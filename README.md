# 多 Agent 多模态医疗辅助问诊

基于 `FastAPI + Gradio + LangGraph + ChromaDB + Ollama/API` 构建的多模态医疗辅助问诊系统。

> **声明：** 本项目仅用于学习和演示，不能替代医生诊断，不提供处方，不构成医疗建议。

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境

复制并编辑 `.env` 文件：

```bash
cp .env.example .env
# 编辑 .env，填写 API Key（如使用 API 模式）
```

编辑 `configs/config.yaml`，选择模型调用方式：

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
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
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
│   ├── agents/         # LangGraph 多 Agent 工作流
│   ├── clients/        # 模型客户端（Ollama / API）
│   ├── rag/            # RAG 知识库（ChromaDB）
│   ├── rules/          # 急症规则引擎 & 安全过滤
│   ├── schemas/        # Pydantic 数据模型
│   ├── services/       # 业务服务层
│   ├── api/            # FastAPI 路由
│   ├── config.py       # 配置加载
│   └── main.py         # FastAPI 入口
├── frontend/
│   └── gradio_app.py   # Gradio 前端
├── data/
│   ├── medical_docs/   # 15 个医学知识文档
│   ├── emergency_rules.json
│   └── department_mapping.json
├── scripts/
│   ├── ingest_docs.py  # 文档入库
│   └── test_query.py   # 检索测试
├── tests/              # 单元测试
└── configs/
    └── config.yaml     # 主配置文件
```

---

## 工作流

```
用户输入 → 输入校验 → 图像分析 → 症状抽取 → 急症判断
                                                    ↓
                            急症路径 ← 是 ← 存在急症 → 否 → 知识检索 → 风险评估 → 建议生成
                                ↓                                                      ↓
                            安全审查 ←──────────────────────────────────────────────────
                                ↓
                            报告生成 → 返回结果
```

---

## 运行测试

```bash
# 测试急症规则引擎
pytest tests/test_rules.py -v

# 测试文档加载
pytest tests/test_rag.py -v

# 测试 Agent 节点
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
| Embedding | bge-small-zh-v1.5 |
| 本地模型 | Ollama (qwen2.5:7b) |
| API 模型 | OpenAI-Compatible API |
