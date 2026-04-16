# 架构说明（MVP）

## 产品目标
为员工提供企业知识问答，确保结果具备：
- 最新性
- 正确性
- 权限可访问性

## 核心流程
1. 用户携带画像信息（`role`、`department`、`project`）发起提问。
2. 意图分类器识别问题类型（制度 / 项目 / FAQ / 推荐请求）。
3. 在最终排序前先做权限过滤，缩小候选知识范围。
4. 检索与重排选出高相关片段，并优先最新版本。
5. 答案编排层输出：
   - 答案摘要
   - 引用来源
   - 版本与生效时间
   - 适用范围与适用人群
6. 记录检索与点击行为，用于后续推送与推荐。

## 服务边界
- `ingest_pipeline`：解析、切分、摘要、打标签并写入索引与元数据。
- `query_pipeline`：执行权限感知检索与答案编排。
- `recommendation_service`：基于角色/部门/项目规则进行推送。
- `feedback_service`：采集“有帮助/过期”等反馈标签。

## 存储
- 向量索引：Chroma（`storage/vector/`）
- 元数据数据库：`storage/sqlite/metadata.db`（表：`knowledge_metadata`）
- 固定夹具数据：`data/fixtures/*.json`
