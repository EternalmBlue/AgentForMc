# gRPC Bridge

后端通过 `AgentBridgeService` 与 Agent4Minecraft 插件通信。

## 启动入口

```powershell
python main.py
```

等价模块入口：

```powershell
python -m agent_for_mc.interfaces.grpc
```

核心文件：

```text
agent_for_mc/interfaces/grpc/server.py
agent_for_mc/interfaces/grpc/service.py
agent_for_mc/interfaces/grpc/runtime.py
agent_for_mc/interfaces/grpc/agent_bridge.proto
```

## 服务定义

```proto
service AgentBridgeService {
  rpc Probe(ProbeRequest) returns (ProbeResponse);
  rpc Ask(AskRequest) returns (AskResponse);
  rpc PrepareSync(SyncPrepareRequest) returns (SyncPrepareResponse);
  rpc UploadFileChunk(stream FileChunkUploadRequest) returns (FileChunkUploadResponse);
  rpc CommitSync(SyncCommitRequest) returns (SyncCommitResponse);
  rpc GetSyncStatus(SyncStatusRequest) returns (SyncStatusResponse);
}
```

## 认证

除 `Probe` 外，所有业务 RPC 都要求：

```text
authorization: Bearer <RAG_GRPC_AUTH_TOKEN>
```

缺失、格式错误或 token 不一致时返回：

```text
UNAUTHENTICATED
```

## Probe

用途：

- 后端存活检测。
- 协议版本检查。
- `server.id` 和 `server_instance_id` 绑定校验。

返回：

- `backend_name = AgentForMc`
- `backend_version`
- `protocol_version`
- `ack`
- `message`

## Ask

接收玩家问题，返回答案。

关键字段：

- `server_id`
- `server_instance_id`
- `player_id`
- `player_name`
- `question`
- `request_id`
- `installed_plugins`

## PrepareSync

接收 manifest，返回后端需要上传的路径。

后端校验：

- 相对路径不为空。
- 路径不包含 `..`、空段、盘符。
- 只允许插件配置和核心服务端配置。
- SHA-256 是 64 位小写 hex。
- size 不为负数。
- manifest 不重复。

## UploadFileChunk

client-streaming 上传单个文件。

后端校验：

- stream 不为空。
- `sync_id` 不变。
- `relative_path` 不变。
- `total_chunks` 不变。
- `sha256` 不变。
- chunk 顺序连续。
- 最终 SHA-256 匹配。

## CommitSync

校验本次同步已上传完整，然后触发语义刷新。

如果没有变更文件，可能直接完成。

## GetSyncStatus

返回：

- 本次同步状态。
- 已接收文件数。
- 已索引文件数。
- 上传字节数。
- 当前上传路径。
- 语义刷新 bundle 总数和完成数。
- 当前刷新 bundle 和 phase。
- 状态消息。

## 状态码映射

| 异常 | gRPC 状态 |
| --- | --- |
| `InvalidRequestError` | `INVALID_ARGUMENT` |
| `NotFoundError` | `NOT_FOUND` |
| `ConflictError` | `ALREADY_EXISTS` |
| `FailedPreconditionError` | `FAILED_PRECONDITION` |
| `BridgeRuntimeError` | `INTERNAL` |

## 协议变更

改动 proto 后需要同步：

- 后端 proto。
- 后端生成代码。
- 插件端 proto。
- 插件端构建。
- 两边测试。
