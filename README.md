# AFAC 金融长文本 Agent

AAADogs 队参加 AFAC2026 挑战组赛题四“金融长文本 Agent 的动态记忆压缩与高效问答挑战”的开源实现。赛题与数据请见[天池官方赛事页](https://tianchi.aliyun.com/competition/entrance/532486)。

## 方法

系统以题级动态记忆为核心，将长文档中的候选证据压缩为当前问题所需的工作窗口。

```text
原始 PDF / HTML
  → 文本、页码与表格结构解析
  → 文档身份与长期证据构建
  → 身份路由 + BM25 候选召回
  → Qwen 文档消歧（按需）
  → 题级证据窗口组装
  → Qwen 判答或证据驱动计算
  → 答案、推理、证据与 Token 记录
```

主要实现包括：

- 中文单字与相邻双字 BM25，对数字、百分数和常见金融写法做查询侧归一。
- 题干与选项分别检索，合并时保护核心命中并兼顾不同来源。
- 保险条款按原文结构建立可定位卡片，财务报表保留单元格的期间、主体和列口径。
- 每道题独立选择文档、组装证据和调用 Qwen；证据不足时扩大词法检索范围。
- 计算题由 Qwen 基于引用证据完成取数与计算，代码负责答案槽位、格式校验和回退流程。
- 运行产物记录模型响应、API Token 用量、证据片段及最终结果，供一致性检查。

## 环境

- Python 3.10+
- DashScope API
- Qwen3.5、Qwen3.6 或 Qwen3.7 系列模型

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export DASHSCOPE_API_KEY="..."
```

也可以复制 `.env.example` 为 `.env` 后填写密钥。

## 数据准备

从有权来源取得赛题材料后，按以下结构放置：

```text
dataset/raw/
├── insurance/
├── financial_contracts/
├── financial_reports/
├── research/
└── regulatory/
    ├── txt/
    ├── html/
    └── attachments/
```

首次生成本地解析结果：

```bash
python script/rebuild_processed.py \
  --input dataset/raw \
  --output processed_data
```

`processed_data` 由运行器直接读取。构建脚本要求输出目录尚不存在。

## 运行

准备题目文件与提交模板后执行：

```bash
./generate_answer.sh \
  --input /path/to/authorized/input \
  --output ./output/run
```

`--input` 可以指向 JSON/JSONL 题目目录，也可以指向同时包含 `questions/` 与 `submit.csv` 的目录。输出目录必须尚不存在。

离线检查：

```bash
./generate_answer.sh --check-runtime
python -m unittest discover -s tests
```

## 代码结构

```text
agent/       检索、文档选择、动态证据、问答与输出格式
script/      文档预处理、证据组装与复现检查
config/      冻结运行配置
tests/       离线单元测试
```

## 团队与联系

- 队伍：AAADogs
- 成员：宋金洲、李翰卿、周帅宇
- 邮箱：ginlater@gmail.com
- 微信：ginlater0515

## 许可

源代码采用 [GNU Affero General Public License v3.0](LICENSE)。赛题数据及第三方内容不包含在本仓库中，其使用条件见 [DATA_AND_CONTENT_LICENSE.md](DATA_AND_CONTENT_LICENSE.md)；依赖许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

感谢赛事主办方、天池平台、Qwen、DashScope 及本项目使用的开源软件作者。
