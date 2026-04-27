# 故障排查

## Missing environment variable RAG_DEEPSEEK_API_KEY

原因：后端没有读取到 DeepSeek API Key。

处理：

```dotenv
RAG_DEEPSEEK_API_KEY=你的DeepSeekAPIKey
```

确认 `.env` 位于仓库根目录，或 `RAG_CONFIG_TOML` / 启动路径没有让你误判配置位置。

## Missing embedding API key

原因：后端没有读取到智谱 embedding API Key。

处理：

```dotenv
RAG_ZHIPU_API_KEY=你的智谱APIKey
```

## Missing gRPC auth token

原因：后端启动 gRPC 时要求 token。

处理：

```dotenv
RAG_GRPC_AUTH_TOKEN=change_me_to_a_strong_token
```

插件端 `backend.authToken` 必须一致。

## Lance 数据库目录不存在

默认目录：

```text
data/plugin_docs_vector_db
```

处理：

- 恢复插件文档向量库。
- 或在 `config.toml` 中修改 `[paths].plugin_docs_vector_db_dir`。

公开仓库一般不会包含 `data/`，所以部署时需要单独准备。

## embedding 维度不匹配

原因：`embedding.dimensions` 与 LanceDB 表里的 embedding list size 不一致。

处理：

- 改回原来的 dimensions。
- 或用新维度重建向量库。

## 插件认证失败

后端：

```dotenv
RAG_GRPC_AUTH_TOKEN=abc
```

插件：

```yaml
backend:
  authToken: "abc"
```

必须完全一致。

## server.id conflict

含义：同一个 `server.id` 已绑定另一个 `server_instance_id`。

处理：

- 给当前 Minecraft 服务端换新的 `server.id`。
- 或确认旧实例废弃后，停止后端并清理：

```text
data/server_instance_bindings.json
```

## `/a4m sync` 后一直 INDEXING

可能原因：

- 配置语义 agent 正在处理大量文件。
- DeepSeek API 慢或失败。
- embedding API 慢或失败。
- LanceDB 写入失败。
- 某个 bundle 处理卡住。

处理：

1. 查看后端日志。
2. 用 `/a4m status` 查看 `current_refresh_bundle` 和 `current_refresh_phase`。
3. 降低 `max_files_per_plugin` 或 `max_file_chars`。
4. 检查 API Key 和网络。

## UploadFileChunk sha256 verification failed

原因：

- 上传内容和 manifest hash 不一致。
- 插件端文件在扫描后被修改。
- 客户端实现不符合协议。

处理：

- 重新执行 `/a4m sync`。
- 确认只使用官方插件端。
- 检查服务端配置是否在同步过程中被自动写入。

## gRPC 服务监听失败

可能原因：

- 端口已被占用。
- 无权限监听指定地址。
- `grpc.port` 超出范围。

处理：

```toml
[grpc]
host = "127.0.0.1"
port = 50052
```

同步修改插件端 `backend.port`。
