import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent import answerer, calc, qwen_client, submission_schema
from agent.retrieval import BM25, query_tokens


class SchemaTests(unittest.TestCase):
    def test_explicit_answer_format_takes_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            question_file = Path(directory) / "questions.json"
            question_file.write_text(
                '[{"qid":"demo_format","type":"计算题",'
                '"answer_format":"mcq","question":"示例",'
                '"options":{"A":"甲","B":"乙"}}]',
                encoding="utf-8",
            )
            questions = submission_schema.load_questions(directory)
        self.assertEqual(questions[0]["answer_format"], "mcq")

    def test_schema_and_question_driven_percent_refinement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "submit.csv"
            with template.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["qid", "answer_1", "answer_2"])
                writer.writerow(["summary", "", ""])
                writer.writerow(["demo_001", "90817.43", ""])
            schema = submission_schema.load_schema(template)
            questions = [{
                "qid": "demo_001",
                "question": "若销量持平，全年需求同比增速最接近多少？",
            }]
            self.assertEqual(
                submission_schema.refine_schema_from_questions(
                    schema, questions)["demo_001"],
                ["percent"],
            )
            self.assertEqual(
                submission_schema.fmt_slot("73.186%", "percent"), "73.19%")

    def test_date_formatting(self):
        self.assertEqual(
            submission_schema.fmt_slot("2030-02-03", "date"),
            "2030年2月3日"
        )

    def test_explicit_plain_number_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / "submit.csv"
            template.write_text(
                "qid,answer_1\nsummary,\ndemo_002,999999.99\n",
                encoding="utf-8-sig",
            )
            schema = submission_schema.load_schema(template)
            submission_schema.refine_schema_from_questions(schema, [{
                "qid": "demo_002",
                "question": "计算同比增长率，保留两位小数、不带单位。",
            }])
            self.assertEqual(schema["demo_002"], ["number"])

    def test_question_driven_date_refinement(self):
        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / "submit.csv"
            template.write_text(
                "qid,answer_1\nsummary,\ndemo_003,999999.99\n",
                encoding="utf-8-sig",
            )
            schema = submission_schema.load_schema(template)
            submission_schema.refine_schema_from_questions(schema, [{
                "qid": "demo_003",
                "question": "期满后的次一工作日是哪一天？",
            }])
            self.assertEqual(schema["demo_003"], ["date"])


class RetrievalTests(unittest.TestCase):
    def test_lexical_ranking(self):
        chunks = [
            {"id": "d1#c1", "doc_id": "d1", "page": 1,
             "text": "甲文档记录绿色债券和票面利率。"},
            {"id": "d2#c1", "doc_id": "d2", "page": 1,
             "text": "乙文档记录仓储流程和运输方案。"},
        ]
        hits = BM25(chunks).search("绿色债券利率", k=1)
        self.assertEqual(hits[0][0]["doc_id"], "d1")

    def test_query_only_financial_aliases(self):
        tokens = query_tokens("2026年一季度现金分红下降")
        for token in ("2026", "26", "q1", "派息", "回落"):
            self.assertIn(token, tokens)

    def test_year_alias_improves_lexical_recall(self):
        chunks = [
            {"id": "d1#c1", "doc_id": "d1", "page": 1,
             "text": "26年第一季度现金股利同比回落。"},
            {"id": "d2#c1", "doc_id": "d2", "page": 1,
             "text": "仓储运输安排。"},
        ]
        hits = BM25(chunks).search("2026年Q1现金分红下降", k=1)
        self.assertEqual(hits[0][0]["doc_id"], "d1")


class CalculationSurfaceTests(unittest.TestCase):
    def test_last_answer_line_and_slot_validation(self):
        content = "取数：仅使用示例证据。\n答案: 73.19%\n"
        answer = calc.parse_calc(content)
        self.assertEqual(answer, "73.19%")
        self.assertTrue(calc.valid_calc(answer, ["percent"]))
        self.assertFalse(calc.valid_calc("73.19", ["percent"]))

    def test_ranking_accepts_two_or_more_objects(self):
        self.assertTrue(calc.valid_calc("甲>乙", ["ranking"]))
        self.assertTrue(calc.valid_calc("甲>乙>丙", ["ranking"]))
        self.assertTrue(calc.valid_calc("甲＞乙＞丙＞丁", ["ranking"]))
        self.assertFalse(calc.valid_calc("甲", ["ranking"]))

    def test_calc_fallback_keeps_primary_evidence(self):
        question = {
            "qid": "demo_calc",
            "question": "示例计算题",
            "domain": "research",
            "doc_ids": ["doc_a"],
        }
        calls = [
            ("计算完成。\n补充检索: 示例指标\n答案: 42", "42"),
            ("补充证据仍不足。", ""),
        ]
        with mock.patch.object(
            calc,
            "calc_evidence",
            side_effect=[("首次证据", ["doc_a#c1"]),
                         ("补检证据", ["doc_a#c2"])],
        ), mock.patch.object(calc, "_call", side_effect=calls), \
                mock.patch.dict("os.environ", {"AFAC_CALC_SINGLE": "1"}):
            answer, info = calc.answer_calc(
                question, ["number"], return_info=True
            )
        self.assertEqual(answer, "42")
        self.assertEqual(info["reasoning_stage"], "primary")
        self.assertEqual(info["evidence_ids"], ["doc_a#c1"])


class ReasoningProvenanceTests(unittest.TestCase):
    def test_invalid_followup_keeps_r1_evidence(self):
        question = {
            "qid": "demo_choice",
            "question": "示例选择题",
            "options": {"A": "甲", "B": "乙"},
            "answer_format": "mcq",
            "domain": "research",
            "doc_ids": ["doc_a"],
        }
        responses = [
            ("分析：首次证据支持 A。\n答案: A\n补充检索: 示例条款", "", {}),
            ("补充证据不足，暂不输出答案。", "", {}),
        ]
        log = io.StringIO()
        with mock.patch.object(
            answerer,
            "evidence_block",
            side_effect=[("首次证据", [{"id": "doc_a#c1"}], set()),
                         ("补检证据", [{"id": "doc_a#c2"}], set())],
        ), mock.patch.object(answerer, "chat", side_effect=responses), \
                mock.patch.object(answerer, "SLIM", True):
            answer, info = answerer.answer_question(question, log=log)
        self.assertEqual(answer, "A")
        self.assertEqual(info["reasoning_stage"], "r1")
        self.assertEqual(info["evidence_ids"], ["doc_a#c1"])


class AccountingTests(unittest.TestCase):
    def test_model_family_and_direct_usage(self):
        self.assertTrue(qwen_client.is_allowed_model("qwen3.6-plus"))
        self.assertFalse(qwen_client.is_allowed_model("qwen-plus"))
        ledger = qwen_client.TokenLedger()
        usage = {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "total_tokens": 150,
        }
        ledger.add("demo_004", "qwen3.6-plus", usage, "unit")
        self.assertEqual(ledger.per_qid["demo_004"], [120, 30])
        self.assertEqual(ledger.totals(), (120, 30, 150))

    def test_usage_requires_one_question(self):
        ledger = qwen_client.TokenLedger()
        with self.assertRaises(RuntimeError):
            ledger.add("", "qwen3.6-plus", {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            })

    def test_error_message_redacts_credentials(self):
        leaked = "sk-" + "exampleCredential123456"
        message = qwen_client._redact_error_message(
            f"Authorization: Bearer {leaked} "
            "api_key=another-secret-value"
        )
        self.assertNotIn(leaked, message)
        self.assertNotIn("another-secret-value", message)
        self.assertEqual(message.count("[REDACTED]"), 2)


if __name__ == "__main__":
    unittest.main()
