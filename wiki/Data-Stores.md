# 数据目录与向量库

## 目录概览

| 路径 | 说明 |
| --- | --- |
| `data/plugin_docs_vector_db` | 插件文档 LanceDB 向量库 |
| `data/server_config_semantic_vector_db` | 上传配置抽取后的语义记忆向量库 |
| `data/user_semantic_memory.sqlite3` | 可选玩家长期记忆 |
| `data/server_instance_bindings.json` | 服务端身份绑定 |
| `mc_servers/<server.id>/...` | 插件上传后的 Minecraft 配置 |
| `.cache/grpc_uploads` | 上传临时文件 |
| `.cache/models` | Hugging Face / Transformers / reranker 缓存 |

## 插件文档向量库

默认路径：

```text
data/plugin_docs_vector_db
```

默认表名：

```text
plugin_docs
```

必需字段：

| 字段 | 说明 |
| --- | --- |
| `id` | 文档 ID |
| `content` | 文档正文 |
| `plugin_chinese_name` | 插件中文名 |
| `plugin_english_name` | 插件英文名 |
| `embedding` | fixed size list embedding |

启动时会校验：

- 数据库目录存在。
- 表能打开。
- 必需字段存在。
- `embedding` 是 fixed size list。
- embedding 维度等于配置的 `embedding.dimensions`。

## BM25 索引

如果启用：

```toml
[plugin_docs_store]
bm25_enabled = true
bm25_auto_create_index = true
```

后端会为 `content` 字段创建 FTS 索引，用于 BM25 检索。

## 上传配置存储

插件上传的配置会保存到：

```text
mc_servers/<server.id>/...
```

例如：

```text
mc_servers/lobby-1/server.properties
mc_servers/lobby-1/plugins/Essentials/config.yml
```

这些文件是后端语义刷新输入。

## 配置语义记忆

默认路径：

```text
data/server_config_semantic_vector_db
```

后端会把上传的配置按插件或 server core 组织成 bundle，调用配置语义抽取 agent，生成 `SemanticMemoryEntry` 并写入 LanceDB。

## 用户长期记忆

启用后：

```toml
[memory]
enabled = true
```

默认存储：

```text
data/user_semantic_memory.sqlite3
```

记忆作用域通常基于：

```text
<server.id>:<player_id>
```

## 不应提交到 Git

以下内容通常不应提交：

- `.env`
- `data/`
- `.cache/`
- `mc_servers/`
- `.venv/`
- `__pycache__/`
- `.pytest_cache/`

公开发布时，建议单独说明如何准备 `data/plugin_docs_vector_db`。
