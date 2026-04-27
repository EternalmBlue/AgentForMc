# 发布 Wiki

GitHub Wiki 是独立 Git 仓库，地址通常是：

```text
https://github.com/EternalmBlue/AgentForMc.wiki.git
```

本目录 `wiki/` 中的文件可以直接复制到 Wiki 仓库根目录。

## 第一次启用 Wiki

如果克隆失败，先在 GitHub 页面创建一次 Wiki：

1. 打开 `https://github.com/EternalmBlue/AgentForMc`
2. 点击 `Wiki`
3. 创建临时 `Home` 页面
4. 保存后再克隆 `.wiki.git`

## 发布命令

```powershell
git clone https://github.com/EternalmBlue/AgentForMc.wiki.git AgentForMc.wiki
Copy-Item F:\AgentForMc\wiki\*.md .\AgentForMc.wiki\ -Force
cd AgentForMc.wiki
git status
git add .
git commit -m "docs: add AgentForMc wiki"
git push origin HEAD
```

## 更新 Wiki

```powershell
cd AgentForMc.wiki
git pull
Copy-Item F:\AgentForMc\wiki\*.md .\ -Force
git add .
git commit -m "docs: update AgentForMc wiki"
git push origin HEAD
```

## 页面说明

- `Home.md` 是 Wiki 首页。
- `_Sidebar.md` 是侧边栏。
- 其他 `.md` 文件会成为独立页面。
- 文件名建议英文，页面标题可以中文。

## 发布前检查

```powershell
Get-ChildItem F:\AgentForMc\wiki
```

确认只复制 Markdown 文件，不复制：

- `.env`
- `data/`
- `.cache/`
- `mc_servers/`
- `.venv/`
- `__pycache__/`
