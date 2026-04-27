# 后端配置

AgentForMc 配置来自两处：

- `.env`：密钥和敏感值。
- `config.toml`：非敏感运行时配置。

可以通过 `RAG_CONFIG_TOML` 指定其他 TOML 文件。

## .env

模板：

```text
.env.example
```

必需变量：

| 变量 | 说明 |
| --- | --- |
| `RAG_ZHIPU_API_KEY` | 智谱 embedding API Key |
| `RAG_DEEPSEEK_API_KEY` | DeepSeek Chat API Key |
| `RAG_GRPC_AUTH_TOKEN` | 插件调用 gRPC 的 Bearer token |

可选变量：

| 变量 | 说明 |
| --- | --- |
| `RAG_CONFIG_TOML` | 指定配置文件路径 |
| `RAG_LANGSMITH_API_KEY` | LangSmith API Key |
| `RAG_OTEL_EXPORTER_OTLP_HEADERS` | OTLP headers |

## gRPC

```toml
[grpc]
host = "127.0.0.1"
port = 50051
max_workers = 8
session_ttl_seconds = 1800
sync_ttl_seconds = 3600
upload_tmp_dir = ".cache/grpc_uploads"
```

跨机器访问：

```toml
[grpc]
host = "0.0.0.0"
port = 50051
```

## embedding

```toml
[embedding]
url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
model = "embedding-3"
dimensions = 1024
```

支持维度：

```text
256
512
1024
2048
```

修改维度后必须重建已有向量表。

## DeepSeek

```toml
[deepseek]
model = "deepseek-chat"
chat_url = "https://api.deepseek.com/chat/completions"
```

## 插件文档向量库

```toml
[paths]
plugin_docs_vector_db_dir = "data/plugin_docs_vector_db"

[plugin_docs_store]
table_name = "plugin_docs"
retrieval_top_k = 5
answer_top_k = 4
citation_preview_chars = 200
bm25_enabled = true
bm25_top_k = 7
bm25_auto_create_index = true
```

## 上传配置语义记忆

```toml
[plugin_semantic_agent]
mc_servers_root = "mc_servers"
scan_on_startup = true
refresh_interval_seconds = 1800
max_file_chars = 12000
max_files_per_plugin = 20

[server_config_semantic_store]
db_dir = "data/server_config_semantic_vector_db"
table_name = "server_config_semantic_memories"
top_k = 8
preview_chars = 220
```

## 长期记忆

默认关闭：

```toml
[memory]
enabled = false
db_path = "data/user_semantic_memory.sqlite3"
recall_limit = 5
min_confidence = 0.75
consolidation_turns = 4
```

## reranker

```toml
[reranker]
enabled = true
model_name_or_path = "maidalun1020/bce-reranker-base_v1"
```

首次启用会下载模型到模型缓存目录。

## 服务端身份绑定

```toml
[server_identity]
bindings_path = "data/server_instance_bindings.json"
```

该文件记录 `server.id` 到 `server_instance_id` 的绑定。
