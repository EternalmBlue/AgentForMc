# 配置同步与语义记忆

本页描述后端如何处理 `/a4m sync` 上传来的 Minecraft 配置文件。

## 接收范围

后端只接受插件允许范围内的相对路径：

- `plugins/` 下的 `.yml` / `.yaml` / `.json` / `.properties` / `.txt` / `.md`
- 根目录下的 `server.properties`
- 根目录下的 `bukkit.yml`
- 根目录下的 `spigot.yml`
- 根目录下的 `paper*.yml`

## PrepareSync

输入：

- `server_id`
- `server_instance_id`
- manifest entries

处理：

1. 校验服务端身份。
2. 校验路径和 SHA-256。
3. 拒绝重复路径和非法路径。
4. 和 `mc_servers/<server.id>/...` 已有文件比较。
5. 返回需要上传的 `required_paths`。

## UploadFileChunk

每个文件单独上传，使用 client-streaming。

处理：

1. 创建 `.cache/grpc_uploads/<sync_id>/` 临时目录。
2. 按顺序写入 chunk。
3. 计算 SHA-256。
4. 校验总块数和 hash。
5. 原子替换到目标路径。

目标路径：

```text
mc_servers/<server.id>/<relative_path>
```

## CommitSync

处理：

1. 校验 `sync_id` 存在。
2. 校验 `server_id` 和 `server_instance_id` 与同步操作一致。
3. 校验提交路径与已成功上传路径一致。
4. 如果有变更文件，启动增量语义刷新。
5. 返回 accepted / indexed / refresh_started。

## GetSyncStatus

返回本地上传和语义刷新状态：

- `PENDING`
- `UPLOADING`
- `INDEXING`
- `COMPLETED`
- `FAILED`

还会返回：

- required file count
- uploaded file count
- total upload bytes
- uploaded bytes
- current upload path
- refresh bundle count
- current refresh bundle
- current refresh phase

## 语义刷新

刷新服务会扫描：

```text
mc_servers/
```

按 server 和 plugin 切分 bundle。核心服务端配置使用特殊 bundle。

每个 bundle 会：

1. 读取允许的文本配置。
2. 截断到 `max_file_chars`。
3. 调用 plugin semantic agent 抽取稳定语义记忆。
4. 规范化和去重。
5. 生成 embedding。
6. 写入 `server_config_semantic_vector_db`。
7. 更新 manifest fingerprint。

## 增量刷新

刷新会比较 manifest fingerprint：

- 未变化的 bundle 跳过。
- 变化的 bundle 重新抽取。
- 已删除的 bundle 从语义记忆中移除。

## 配置项

```toml
[plugin_semantic_agent]
mc_servers_root = "mc_servers"
refresh_interval_seconds = 1800
max_file_chars = 12000
max_files_per_plugin = 20

[server_config_semantic_store]
db_dir = "data/server_config_semantic_vector_db"
table_name = "server_config_semantic_memories"
top_k = 8
preview_chars = 220
```

## 隐私说明

插件端会先做上传前脱敏，但后端默认信任客户端上传内容。非官方或旧版客户端应自行保证上传前脱敏。
