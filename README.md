# 知达智能体演示项目

该智能知识Agent面向企业办公场景下知识管理分散、获取被动等核心问题，打造可精准解答知识疑问、主动推送适配场景信息的智能知识助手。实现打破信息孤岛、激活知识价值、全面提升团队协作效率的核心价值。
当前仍处于开发版本，核心聚焦：
- 权限感知检索
- 版本感知回答
- 个性化分发


## 项目结构
- `app/`：后端 API、领域服务、流程编排。
- `data/`：原始样本、处理产物、固定夹具数据。
- `scripts/`：数据准备和入库脚本。
- `docs/`：架构与交付计划。
- `frontend/web/`：前端页面占位目录。
- `bot/feishu/`：飞书机器人占位目录。
- `tests/`：单元与集成测试。

## 快速启动
```
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
python scripts/demo_cli.py
```

## 入库与查询
```
# 执行知识入库（读取 data/fixtures 下的 .txt 文档）
python scripts/ingest_knowledge.py --source-dir data/fixtures
```

## CLI 端到端演示
```bash
python scripts/demo_cli.py
```

CLI 输入格式固定为：
`问题：***。职级：***。部门：***。`

示例：
`问题：出差报销最新标准是什么？。职级：普通员工。部门：人力。`

输入后可在 CLI 内继续测试：
- `click <引用序号>`：记录引用点击
- `feedback`：提交有帮助/过期反馈
- `recommend`：查看推荐结果
- `push`：触发手动推送
- `next`：继续下一问
- `exit`：退出

向量索引默认使用 Chroma，落盘目录为 `storage/vector/`。
Embedding 默认使用 DashScope `text-embedding-v4`，请先配置 `DASHSCOPE_API_KEY`。

## LLM 配置
问答主链路中的意图解析、主答案生成、版本差异总结默认使用 `qwen3-max`。

```bash
# .env
FEISHU_BASE_URL=https://open.feishu.cn
FEISHU_APP_ID=你的飞书AppID
FEISHU_APP_SECRET=你的飞书AppSecret
# 可选：直接配置可用 token，配置后会跳过 app_id/app_secret 换 token
FEISHU_TENANT_ACCESS_TOKEN=
FEISHU_USER_ID_TYPE=open_id
FEISHU_DEPARTMENT_ID_TYPE=open_department_id

CHAT_MODEL=qwen3-max
ANSWER_MODEL=qwen3-max
ENABLE_LLM_ANSWER_GENERATION=true
VERSION_DIFF_MODEL=qwen3-max
ENABLE_LLM_VERSION_DIFF=true
CHAT_CONCLUSION_MODEL=qwen3-max
ENABLE_LLM_CHAT_CONCLUSION=false
RERANK_MODEL=gte-rerank
ENABLE_MODEL_RERANK=false
DASHSCOPE_API_KEY=你的Key
OPENAI_BASE_URL=
```

说明：提问时身份识别会优先调用飞书通讯录 API 获取用户与部门信息（项目从问题中提取，未提及则为 `unknown`）。
`IntentParserService` 会优先使用 DashScope 兼容端点调用 `qwen3-max`。
聊天类知识入库前会自动做“聊天结论提炼”（规则默认开启，LLM 提炼可选开启）。
主答案生成会将 rerank 返回的引用片段输入模型，并强制检查“版本/引用”字段。
检索阶段支持可选 rerank 模型，默认 `gte-rerank`，并带规则兜底重排。
命中旧版本时，版本差异默认优先使用 LLM 总结，不可用时回退规则对比。

## Query 工作流（LangGraph）
`/query` 主链路已通过 LangGraph 编排以下节点：
1. `identity`：实时获取用户身份（飞书用户与部门 + 问题内项目提取）
2. `intent`：解析意图与检索 query（LLM 优先，规则兜底）
3. `retrieve`：权限前置检索 + 兜底过滤
4. `version`：版本提示构建
5. `answer`：结构化答案生成（制度类模板/通用模板）

当本地未安装 `langgraph` 时，会自动回退为同逻辑的顺序执行。

## 推荐与推送接口
- `POST /recommendation`：基于用户画像 + 最近查询/点击行为 + 文档新鲜度返回结构化推荐。
- `POST /recommendation/push/trigger`：手动触发推荐推送并落推送日志（`recommendation_push_logs`）。

`/recommendation` 返回示例：
```json
{
  "items": [
    {
      "doc_id": "fixture-02",
      "title": "字节跳动员工差旅与报销制度（V2.0）",
      "source_type": "policy",
      "version": "V2.0",
      "updated_at": "2026-03-15T00:00:00",
      "summary": "报销需在出差结束后10个自然日内发起...",
      "score": 4.2,
      "reasons": ["角色/部门/项目权限匹配", "文档近期更新", "匹配最近搜索主题"]
    }
  ]
}
```


## 演示推荐输入
由于当前尚未接入飞书，且暂无真实企业知识文档，该项目暂且使用模拟知识文档
注：该项目中知识库均为演示模拟所建，其中所涉及的公司名、项目名等，与实际字节跳动公司、项目无关

1.演示权限差异（普通员工 vs 财务）
    问题：同一出差行程出现多张相近时间的打车票如何处理？。职级：普通员工。部门：人力。
    问题：同一出差行程出现多张相近时间的打车票如何处理？。职级：财务专员。部门：财务。


2.权限差异（同职级不同部门）
    问题：项目北极星的交付节点是什么？。职级：项目经理。部门：人力。
    问题：项目北极星的交付节点是什么？。职级：项目经理。部门：交付。

3.旧版命中提醒（版本感知）
    问题：字节跳动员工差旅与报销制度V1.0里报销时限是多少？。职级：普通员工。部门：人力。
    问题：出差报销最新标准是什么？。职级：普通员工。部门：人力。


4.推荐/推送效果（先造行为再展示）
    问题：报销超时提交如何处理？。职级：财务专员。部门：财务。
    问题：哪些情况需要人工复核报销单？。职级：财务专员。部门：财务。

    每条回答后在 CLI 输入：
    click 1

    然后输入：
    recommend
    push

