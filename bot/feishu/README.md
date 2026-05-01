# 飞书 Bot 占位说明

计划能力：
- 接收来自飞书的问题
- 从飞书事件中提取 `open_id/user_id`
- 调用后端 `/query` API，并透传 `user_id_type`
- 返回结构化答案、引用来源和版本提示

当前后端已支持：

- 使用 `lark-oapi` 查询用户部门、岗位、职级
- 根据 `department/title/level/role` 做权限感知检索
- 飞书联系人调用失败时可回退本地 mock 目录
- 提供飞书事件回调入口，默认路径 `/feishu/events`
- 收到文本消息后自动调用知识问答主链路并回发答案

建议 `.env` 至少配置：

- `FEISHU_BOT_ENABLED=true`
- `FEISHU_CONTACT_API_ENABLED=true`
- `FEISHU_APP_ID=...`
- `FEISHU_APP_SECRET=...`
- `FEISHU_VERIFICATION_TOKEN=...`

说明：

- `FEISHU_APP_ID` 和 `FEISHU_APP_SECRET` 是必填项，否则无法调用联系人 API，也无法向飞书发回消息
- `FEISHU_VERIFICATION_TOKEN` 建议与飞书事件订阅配置保持一致
- `FEISHU_ENCRYPT_KEY` 目前只预留配置，当前代码尚未做事件解密
