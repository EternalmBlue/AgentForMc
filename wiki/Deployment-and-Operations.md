# 部署与运维

## 本地部署

默认适合后端和 Minecraft 服务端同机：

```toml
[grpc]
host = "127.0.0.1"
port = 50051
```

插件端无需配置 host / port，只配置 token。

## 跨机器部署

后端：

```toml
[grpc]
host = "0.0.0.0"
port = 50051
```

插件：

```yaml
backend:
  authToken: "change_me_to_a_strong_token"
  host: "后端机器IP"
  port: 50051
```

建议：

- 使用防火墙限制 Minecraft 服务端 IP。
- 不要直接暴露给公网。
- 生产环境评估 TLS / mTLS。

## 启动顺序

推荐：

1. 启动 AgentForMc。
2. 确认 gRPC listening 日志。
3. 启动 Minecraft 服务端。
4. 插件执行 Probe。
5. Probe 成功后插件启用。

如果先启动 Minecraft 服务端，后端不可用会导致插件启动探测失败并自禁用。修复后需要重启服务端或重新加载插件。

## 关键日志

后端启动失败常见日志：

- missing DeepSeek API key
- missing embedding API key
- missing gRPC auth token
- Lance 数据库目录不存在
- embedding 维度不匹配
- gRPC 服务监听失败

插件侧会显示：

- attempted backend endpoint
- server identity
- plugin protocol
- backend response
- failure detail

## 数据备份

建议备份：

- `.env`
- `config.toml`
- `data/plugin_docs_vector_db`
- `data/server_config_semantic_vector_db`
- `data/user_semantic_memory.sqlite3`
- `data/server_instance_bindings.json`
- `mc_servers/`

不要把备份提交到 GitHub。

## 模型缓存

reranker 或相关模型会使用：

```text
.cache/models
```

冷启动时可能下载模型，生产环境可以提前预热。

## token 轮换

1. 停止 Minecraft 服务端。
2. 停止后端。
3. 修改后端 `.env` 的 `RAG_GRPC_AUTH_TOKEN`。
4. 修改插件 `config.yml` 的 `backend.authToken`。
5. 启动后端。
6. 启动 Minecraft 服务端。

## server.id 迁移

如果迁移 Minecraft 服务端：

- 尽量复制插件数据目录，保留 `server-instance-id.txt`。
- 如果是新物理实例，使用新的 `server.id`。
- 只有确认旧实例废弃时，才清理后端绑定文件。

## 发布前检查

- `pytest` 通过。
- 后端能启动。
- 插件 Probe 成功。
- `/a4m sync` 成功。
- `/askmc` 有答案。
- `.env` 未提交。
- `data/`、`.cache/`、`mc_servers/` 未提交。
