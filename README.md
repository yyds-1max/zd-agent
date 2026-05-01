# 知达Agent

面向企业用户的智能知识分发助手。项目聚焦企业内部知识问答场景，围绕“查得到、答得准、看得到最新版本、且符合权限边界”来设计主链路。

当前仓库已具备一个可运行的后端原型，支持本地样例知识库、权限感知检索、多轮会话理解、版本校验与飞书 Bot 回调。

## 项目定位

企业内部知识问答和知识分发，通常会遇到几个核心问题：

- 同一主题存在多个版本，员工容易看到旧制度
- 不同角色、部门、项目成员看到的内容范围不同
- 用户提问常常是口语化追问，而不是标准搜索词
- 知识分散在制度、FAQ、项目资料、会议结论中，检索路径不统一

知达Agent 当前的实现重点，就是把这些问题放进一条统一的问答链路里处理。

## 当前能力

- 权限感知检索
  - 检索前先根据 `role / department / title / level / project` 做访问范围过滤
  - 支持飞书联系人信息与本地用户目录双来源
- 多源知识问答
  - 样例数据覆盖制度、项目资料、FAQ、聊天结论等类型
  - 文档会被切分为 chunk，并继承版本、主题、权限等元数据
- 混合检索与重排
  - 使用 BM25 + Chroma 向量检索混合召回
  - 通过 RRF 融合结果，并支持 `gte-rerank-v2` 重排
  - 当依赖或密钥不可用时，可自动回退到本地近似能力
- 版本感知回答
  - 支持按 `status / version / published_at` 判断推荐版本
  - 当用户命中旧版内容时，可补充版本提醒和差异说明
- 多轮会话理解
  - 支持基于历史上下文改写追问
  - 会话记录默认落到本地 JSON 文件，便于 Demo 和调试
- Agent 化编排
  - 主流程由主 Agent 协调画像、检索、版本检查、差异分析和答案生成
  - 优先使用 LangGraph 风格编排，便于后续扩展成更复杂的工作流
- 飞书 Bot 接入
  - 支持飞书事件回调入口
  - 支持基于消息发送人的 `open_id / user_id` 做权限感知问答
  - 支持“新建会话”菜单事件，便于重置上下文

## 适合演示的典型场景

- 员工提问“出差报销最新标准是什么”，系统优先返回当前生效制度并附带引用
- 财务、项目经理、普通员工提问同一问题，返回结果因权限不同而不同
- 用户连续追问“那销售部门适用吗？”，系统结合上一轮上下文理解问题
- 用户命中旧版制度内容时，系统提示当前推荐版本，并给出变更摘要
- 飞书 Bot 收到企业内部问题后，直接复用后端问答链路返回答案

<<<<<<< HEAD
## 快速启动
```
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
python scripts/demo_cli.py
```

## 入库与查询
=======
## 核心流程

```mermaid
flowchart LR
    A[用户提问] --> B[用户画像与意图识别]
    B --> C[权限过滤]
    C --> D[BM25 + 向量混合检索]
    D --> E[RRF 融合与重排]
    E --> F[版本检查 / 差异分析]
    F --> G[答案生成]
    G --> H[引用与版本提示返回]
```

## 技术栈

- 后端框架：FastAPI
- 模型编排：LangChain、LangGraph
- 向量存储：Chroma
- 检索策略：BM25 + 向量检索 + RRF + Rerank
- LLM：DashScope 兼容接口，默认 `qwen3-max`
- Embedding：默认 `text-embedding-v4`
- Rerank：默认 `gte-rerank-v2`
- 外部集成：飞书开放平台 `lark-oapi`
- 测试：Pytest

## 已实现模块

- `QueryPipeline`
  - 统一处理会话上下文、问题改写和主问答链路
- `KnowledgeDispatchMainAgent`
  - 负责意图判断、工具协调和最终回答生成
- `UserProfileTool`
  - 负责用户画像补全、项目名识别、意图辅助判断
- `KnowledgeRetrievalTool`
  - 负责权限过滤后的知识召回与补充检索
- `LatestVersionTool`
  - 负责当前推荐版本判断
- `VersionDiffTool`
  - 负责旧版与新版差异总结
- 飞书 Bot 服务
  - 负责事件校验、消息接收、回复发送和会话切换

## 仓库结构

```text
.
├── app/                    # 核心后端代码
│   ├── api/                # FastAPI 路由
│   ├── core/               # 配置与基础组件
│   ├── pipelines/          # 主流程编排
│   ├── repositories/       # 数据访问层
│   ├── schemas/            # 数据结构定义
│   └── services/           # 检索、画像、版本、Agent 等服务
├── bot/feishu/             # 飞书 Bot 说明
├── data/                   # 样例知识与 mock 用户目录
├── docs/                   # 架构与路线图文档
├── frontend/web/           # Web 前端占位目录
├── scripts/                # 索引重建与 LLM smoke test 脚本
├── storage/                # 本地持久化数据
└── tests/                  # 单元测试
>>>>>>> feat: refactor query pipeline and update GitHub README
```

## 样例数据

当前仓库已经内置一套可直接演示的样例知识：

- 制度文档：如差旅与报销制度的多个版本
- FAQ：常见办公问题与财务 FAQ
- 项目资料：项目北极星需求说明、周报、交付节点
- 聊天结论：从项目群结论中沉淀出的可检索知识
- 权限与元数据样本：用于演示权限差异和治理规则

Demo 用户也已经内置：

- `u_employee_li`：普通员工
- `u_finance_wang`：财务专员
- `u_pm_zhou`：项目经理
- `u_newhire_chen`：新员工

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

默认情况下，即使不开启真实模型能力，项目也可以基于本地样例数据跑通主流程。

如果希望启用完整能力，建议至少配置：

```env
DASHSCOPE_API_KEY=your_api_key
ENABLE_USER_PROFILE_LLM=true
ENABLE_CONVERSATION_ROUTER_LLM=true
ENABLE_ANSWER_LLM=true
ENABLE_VERSION_DIFF_LLM=true
```

如果需要接入飞书，再补充：

```env
FEISHU_BOT_ENABLED=true
FEISHU_CONTACT_API_ENABLED=true
FEISHU_APP_ID=cli_xxx
FEISHU_APP_SECRET=xxx
FEISHU_VERIFICATION_TOKEN=xxx
```

## 运行方式

### CLI 问答

```bash
python -m app.main --user-id u_employee_li --question "出差报销最新标准是什么？"
```

<<<<<<< HEAD
## LLM 配置
问答主链路中的意图解析、主答案生成、版本差异总结默认使用 `qwen3-max`。
=======
支持多轮会话参数：
>>>>>>> feat: refactor query pipeline and update GitHub README

```bash
python -m app.main \
  --user-id u_employee_li \
  --conversation-id demo-conv-1 \
  --question "那销售部门适用吗？"
```

### 启动 API 服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

接口：

- `POST /query`
- 飞书事件回调默认路径：`POST /feishu/events`

<<<<<<< HEAD
## 推荐与推送接口
- `POST /recommendation`：基于用户画像 + 最近查询/点击行为 + 文档新鲜度返回结构化推荐。
- `POST /recommendation/push/trigger`：手动触发推荐推送并落推送日志（`recommendation_push_logs`）。
=======
### `/query` 请求示例
>>>>>>> feat: refactor query pipeline and update GitHub README

```json
{
  "user_id": "u_employee_li",
  "user_id_type": "open_id",
  "question": "出差报销最新标准是什么？",
  "top_k": 4,
  "conversation_id": "demo-conv-1",
  "use_history": true
}
```

响应中会包含：

- `answer`：最终回答
- `citations`：引用片段
- `version_checks`：版本校验结果
- `version_diffs`：版本差异结果
- `version_notice`：版本提醒
- `tool_trace`：工具链执行轨迹

## 索引重建

当你修改了 `data/fixtures` 中的知识内容后，可以重建 Chroma 索引：

```bash
python scripts/rebuild_index.py
```

如果想保留现有 collection，仅执行 upsert：

```bash
python scripts/rebuild_index.py --keep-existing
```

说明：

- 该脚本依赖 `chromadb`
- 需要可用的 `DASHSCOPE_API_KEY`
- 默认索引路径为 `storage/vector/chroma`

## LLM Smoke Test

可以单独验证 LLM 相关链路：

```bash
python scripts/test_llm_chain.py profile
python scripts/test_llm_chain.py answer
python scripts/test_llm_chain.py both
```

打印完整测试上下文：

```bash
python scripts/test_llm_chain.py answer --print-context
```

## 测试

运行单元测试：

```bash
pytest
```

当前测试已覆盖的重点包括：

<<<<<<< HEAD
## 演示推荐输入
由于当前尚未接入飞书，且暂无真实企业知识文档，该项目暂且使用模拟知识文档

注：该项目中知识库均为演示模拟所建，其中所涉及的公司名、项目名等，与实际字节跳动公司、项目无关
=======
- 问答主链路
- 权限过滤
- 版本判断
- 飞书 Bot 回调与消息处理
- 多轮会话上下文
>>>>>>> feat: refactor query pipeline and update GitHub README

## 当前状态

这是一个偏后端能力验证的原型版本，目前更适合：

- 企业知识助手 Demo
- RAG / Agent 主链路验证
- 飞书知识问答机器人 PoC
- 企业内知识治理产品的早期方案孵化

以下方向已留出结构，但仍有继续完善空间：

- 推荐与订阅触发链路的产品化闭环
- 前端 Web 控制台
- 更完整的知识入库 pipeline
- 反馈治理与运营数据看板
- 生产级权限系统与企业数据源打通

## 相关文档

- [架构说明](docs/ARCHITECTURE.md)
- [路线图](docs/ROADMAP.md)
- [样例数据说明](data/README.md)
- [飞书 Bot 说明](bot/feishu/README.md)

## License

当前仓库未单独声明开源许可证；如果计划公开发布到 GitHub，建议补充 `LICENSE` 文件后再对外分发。
