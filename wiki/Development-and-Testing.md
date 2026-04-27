# 开发与测试

## 安装开发环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 运行测试

```powershell
pytest
```

指定测试：

```powershell
pytest tests/test_grpc_bridge.py
pytest tests/test_plugin_semantic_scanner.py
pytest tests/test_graph_and_tools.py
pytest tests/test_observability.py
```

## 重点测试领域

- gRPC Bearer token 认证。
- `Probe` 免认证但校验身份。
- `server.id` 冲突检测。
- manifest 路径校验。
- 分块上传顺序和 SHA-256。
- `CommitSync` 状态机。
- `GetSyncStatus` 刷新进度。
- 插件配置扫描和 bundle 切分。
- DeepAgent 工具注册。
- 可观测性埋点。

## gRPC 测试

测试中使用本地 gRPC server：

```text
grpc.server(...)
127.0.0.1:0
```

这样可以覆盖真实 service、metadata、状态码和 proto mapping。

## 修改 proto

如果修改：

```text
agent_for_mc/interfaces/grpc/agent_bridge.proto
```

需要重新生成：

```text
agent_for_mc/interfaces/grpc/agent_bridge_pb2.py
agent_for_mc/interfaces/grpc/agent_bridge_pb2_grpc.py
```

同时更新插件仓库：

```text
src/main/proto/agent_bridge.proto
```

并运行两边测试。

## 修改配置加载

需要覆盖：

- 默认值。
- `.env` 变量。
- `config.toml` 覆盖。
- 路径相对 `config.toml` 所在目录解析。
- 启动校验错误信息。

## 修改上传同步

必须验证：

- 非法路径被拒绝。
- 重复路径被拒绝。
- 不需要上传的文件被跳过。
- SHA-256 不匹配会失败。
- commit 路径必须与已上传路径一致。
- 失败后临时目录清理。

## 修改语义刷新

必须验证：

- bundle fingerprint 生效。
- 未变化 bundle 跳过。
- stale bundle 删除。
- agent 输出 JSON 解析和修复逻辑。
- embedding 维度和 store schema 一致。

## 手工联调

1. 启动后端 `python main.py`。
2. 启动 Paper 服务端。
3. 插件启动探测成功。
4. `/a4m sync`
5. `/a4m status`
6. `/askmc <问题>`
7. 关闭后端后验证插件错误展示。
