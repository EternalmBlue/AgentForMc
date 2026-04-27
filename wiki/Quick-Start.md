# 快速开始

本页用于启动 AgentForMc 后端，并与 Agent4Minecraft 插件完成本地联调。

## 前置条件

- Python 3.11+
- DeepSeek API Key
- 智谱 embedding API Key
- 已准备插件文档 LanceDB 向量库
- Agent4Minecraft 插件端

## 克隆仓库

```powershell
git clone https://github.com/EternalmBlue/AgentForMc.git
git clone https://github.com/EternalmBlue/Agent4Minecraft.git
```

## 安装依赖

```powershell
cd AgentForMc
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux / macOS：

```bash
cd AgentForMc
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 配置密钥

```powershell
Copy-Item .env.example .env
```

编辑 `.env`：

```dotenv
RAG_ZHIPU_API_KEY=你的智谱APIKey
RAG_DEEPSEEK_API_KEY=你的DeepSeekAPIKey
RAG_GRPC_AUTH_TOKEN=change_me_to_a_strong_token
```

## 准备插件文档向量库

默认路径：

```text
data/plugin_docs_vector_db
```

默认表名：

```text
plugin_docs
```

该 LanceDB 表必须包含：

```text
id
content
plugin_chinese_name
plugin_english_name
embedding
```

如果该目录不存在，后端启动会失败。

## 启动后端

```powershell
python main.py
```

或：

```powershell
python -m agent_for_mc.interfaces.grpc
```

默认监听：

```text
127.0.0.1:50051
```

## 配置插件端

在 Minecraft 服务端：

```yaml
backend:
  authToken: "change_me_to_a_strong_token"
```

如果后端不在同一台机器：

```yaml
backend:
  authToken: "change_me_to_a_strong_token"
  host: "后端机器IP"
  port: 50051
```

## 联调命令

游戏内执行：

```text
/a4m sync
/a4m status
/askmc eco 插件的金币倍率在哪里配置？
```

## 成功标准

- 后端启动没有配置错误。
- 插件启动探测成功。
- `/a4m sync` 能提交配置。
- `/a4m status` 能看到远程状态。
- `/askmc` 能收到回答。
