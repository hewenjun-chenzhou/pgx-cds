# PGx CDS 启动指南

## 一、双击启动（最简单）

1. 打开文件资源管理器，进入目录 `D:\竞赛项目\项目\pgx_cds`
2. **双击 `start.bat`**
3. 浏览器会自动打开 <http://127.0.0.1:8501>（若没自动开，手动复制这个地址）
4. 关闭弹出的"黑色窗口"即停止服务

## 二、命令行启动（推荐开发时用）

按 `Win + R`，输入 `cmd`，回车，然后：

```cmd
cd /d D:\竞赛项目\项目\pgx_cds
python -m streamlit run app.py
```

- 想看实时日志（含报错）：留在命令行里
- 想后台跑：`start /b python -m streamlit run app.py`
- 停止：在命令行里按 `Ctrl + C`

## 三、停止服务

| 方式 | 怎么做 |
|---|---|
| 双击启动 | 关掉黑色窗口 |
| 命令行启动 | `Ctrl + C` |
| 强制结束 | 双击 `stop.bat` |

## 四、依赖检查（首次运行才需要）

如果报错 `No module named 'streamlit'`，请先安装依赖：

```cmd
cd /d D:\竞赛项目\项目\pgx_cds
python -m pip install -r requirements.txt
```

（如果 `requirements.txt` 不存在，可手动装：`pip install streamlit pandas openai`）

## 五、常见问题

**Q：双击 start.bat 后窗口闪退？**
A：缺 Python 或缺依赖。按"依赖检查"那节装好后重试。

**Q：端口 8501 被占用？**
A：双击 `stop.bat` 杀掉旧进程，或手动改 `start.bat` 里的 `--server.port 8501` 为 8502。

**Q：浏览器没自动开？**
A：手动复制 <http://127.0.0.1:8501> 到浏览器。
