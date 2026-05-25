# 多 Agent 多模态医疗辅助问诊

基于 `FastAPI + Gradio + LangGraph + ChromaDB + Ollama/API` 构建的多模态医疗辅助问诊系统，支持图文混合输入、RAG 知识增强、三链路急症识别与 Qwen2.5-7B 三阶段强化微调。

> **声明：** 本项目仅用于学习和演示，不能替代医生诊断，不提供处方，不构成医疗建议。

---

## 核心特性

- **10 节点 LangGraph 有向图**：输入校验 → 图像分析 → 症状抽取 → 急症判断 → 知识检索 → 风险评估 → 建议生成 → 安全审查全链路，条件路由支持急症与常规双路径并行分流
- **多模态输入**：VLM 提取图片医学体征描述，与文本症状融合后统一进入推理链
- **自研 RAG 检索**：15 篇结构化医学文档经 BAAI/bge-small-zh-v1.5 向量化存入 ChromaDB，余弦相似度 Top-K 检索注入 Prompt
- **三链路急症识别**：关键词规则（O(n) 精确匹配）→ Embedding 语义兜底 → 灰色地带 LLM 二次确认，F1 从 75.0% 提升至 **95.2%**，Recall **100%**
- **三阶段微调**：SFT（QLoRA r=16）→ GRPO（可验证规则奖励）→ DPO（自动构造偏好对）微调 Qwen2.5-7B，风险宽松准确率由 96.6% 提升至 **100%**
- **双模式部署**：本地 Ollama（qwen2.5:7b）与 OpenAI-Compatible API 配置驱动切换

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境

```bash
cp .env.example .env
# 编辑 .env，填写 API Key（使用 API 模式时）
```

编辑 `configs/config.yaml` 选择模型调用方式：

```yaml
llm:
  text_provider: "ollama"   # 或 "api"
  vision_provider: "api"    # 推荐 API（多模态效果更好）
```

### 3. 启动 Ollama（本地模式）

```bash
ollama serve
ollama pull qwen2.5:7b
```

### 4. 构建医学知识库

```bash
python scripts/ingest_docs.py
```

### 5. 启动后端

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 6. 启动前端

```bash
python frontend/gradio_app.py
```

访问：http://localhost:7860

---

## 项目结构

```
medical-consultation-demo/
├── app/
│   ├── agents/
│   │   ├── graph.py            # LangGraph 有向图定义（10 节点）
│   │   ├── nodes.py            # 各节点实现（含三链路急症判断）
│   │   ├── prompts.py          # Prompt 模板
│   │   └── state.py            # ConsultationState 定义
│   ├── clients/                # 模型客户端（Ollama / OpenAI-Compatible API）
│   ├── rag/
│   │   ├── embedding.py        # bge-small-zh-v1.5 向量化
│   │   ├── vector_store.py     # ChromaDB 封装
│   │   ├── document_loader.py  # 医学文档加载
│   │   └── retriever.py        # Top-K 检索
│   ├── rules/
│   │   ├── emergency_engine.py       # 关键词规则引擎
│   │   ├── emergency_embedding.py    # Embedding 语义急症检索
│   │   └── safety_rules.py           # 安全审查过滤
│   ├── schemas/                # Pydantic 数据模型
│   ├── services/               # 业务服务层
│   ├── api/                    # FastAPI 路由
│   ├── config.py               # 配置加载
│   └── main.py                 # FastAPI 入口
├── frontend/
│   └── gradio_app.py           # Gradio Web UI
├── data/
│   ├── medical_docs/           # 15 篇结构化医学知识文档
│   ├── emergency_rules.json    # 关键词急症规则
│   ├── emergency_cases.json    # 急症语义检索案例库（14 类 ~80 条）
│   ├── department_mapping.json # 科室映射
│   ├── test_cases.jsonl        # 30 条标注评估集
│   ├── train_sft.jsonl         # SFT 训练数据
│   ├── train_reward.jsonl      # GRPO 奖励训练数据
│   └── train_dpo.jsonl         # DPO 偏好对数据
├── train/
│   ├── data_gen.py             # 训练数据生成（模板扩充 + LLM 增强）
│   ├── sft.py                  # SFT 监督微调（QLoRA r=16）
│   ├── grpo.py                 # GRPO 强化训练（可验证规则奖励）
│   ├── dpo.py                  # DPO 偏好对齐（自动构造偏好对）
│   ├── merge.py                # LoRA 权重合并
│   └── requirements.txt        # 训练依赖
├── scripts/
│   ├── ingest_docs.py          # 文档向量化入库
│   ├── eval_quality.py         # 系统质量评估（30 条测试集）
│   ├── eval_compare.py         # 微调前后指标对比
│   └── test_query.py           # 检索调试
├── eval/
│   ├── locustfile.py           # Locust 性能压测
│   └── compare_result.json     # 微调前后对比结果
├── tests/                      # 单元测试
└── configs/
    └── config.yaml             # 主配置文件
```

---

## 工作流

```
用户输入（文字 + 可选图片）
        ↓
    输入校验
        ↓
    图像分析（VLM）─── 无图片则跳过
        ↓
    症状抽取
        ↓
    急症判断 ─── 三链路：关键词 → Embedding → LLM 确认
        ↓
   是急症？
   ↙      ↘
急症路径   常规路径
   ↓            ↓
立即预警    知识检索（RAG）
   ↓            ↓
   ↓       风险评估
   ↓            ↓
   ↓       建议生成
    ↘      ↙
     安全审查
        ↓
     返回结果
```

---

## 急症识别优化过程

| 方案 | Precision | Recall | F1 | 问题 |
|---|---|---|---|---|
| 纯关键词规则 | 100% | 60% | 75.0% | 措辞差异导致漏报 |
| + Embedding 兜底（阈值 0.75） | 66.7% | 100% | 80.0% | 误报增多 |
| + 三区间阈值 + LLM 二次确认 | **90.9%** | **100%** | **95.2%** | — |

三区间策略：
- 相似度 ≥ 0.88：直接判定急症
- 0.75 ≤ 相似度 < 0.88：触发 LLM 二次确认
- 相似度 < 0.75：非急症放行

---

## 微调流程

### 环境准备

```bash
pip install -r train/requirements.txt
# 下载基底模型（ModelScope）
python -c "from modelscope import snapshot_download; snapshot_download('qwen/Qwen2.5-7B-Instruct', local_dir='models/Qwen2.5-7B-Instruct')"
```

### 训练步骤

```bash
# Step 1：生成训练数据
python train/data_gen.py              # 模板扩充 180 条
python train/data_gen.py --llm-augment  # 可选：LLM 增强

# Step 2：SFT 监督微调（QLoRA，单卡 16GB 可跑）
python train/sft.py --model models/Qwen2.5-7B-Instruct --output output/sft

# Step 3：GRPO 强化训练
python train/grpo.py --model output/sft --output output/grpo

# Step 4：DPO 偏好对齐
python train/dpo.py --build-pairs --train --model output/grpo --output output/dpo

# Step 5：合并 LoRA 权重
python train/merge.py --base models/Qwen2.5-7B-Instruct --adapter output/dpo --output output/merged
```

### 微调前后对比

| 指标 | Baseline | Fine-tuned | 变化 |
|---|---|---|---|
| 风险严格准确率 | 72.4% | 69.0% | -3.4pp |
| 风险宽松准确率 ±1 | 96.6% | **100%** | +3.4pp |
| 急症 Precision | 90.9% | 90.9% | — |
| 急症 Recall | 100% | **100%** | — |
| 急症 F1 | 95.2% | **95.2%** | — |

---

## 评估

```bash
# 系统质量评估（需后端运行）
python scripts/eval_quality.py

# 微调前后对比（两阶段，避免 VRAM 冲突）
python scripts/eval_compare.py --ft-model output/merged

# 仅评估微调模型（后端不需要运行）
python scripts/eval_compare.py --ft-model output/merged --skip-baseline

# 性能压测
locust -f eval/locustfile.py --headless -u 5 -r 1 --run-time 60s --host http://localhost:8000 --html eval/report.html
```

---

## 运行测试

```bash
pytest tests/test_rules.py -v
pytest tests/test_rag.py -v
pytest tests/test_agents.py -v
```

---

## 技术栈

| 模块 | 技术 |
|---|---|
| 前端 | Gradio |
| 后端 | FastAPI |
| Agent 编排 | LangGraph |
| 向量数据库 | ChromaDB |
| Embedding | BAAI/bge-small-zh-v1.5 |
| 本地模型 | Ollama (qwen2.5:7b) |
| API 模型 | OpenAI-Compatible API |
| 微调框架 | PEFT (QLoRA) + TRL (SFT/GRPO/DPO) |
| 训练加速 | BitsAndBytesConfig (4-bit NF4) |
