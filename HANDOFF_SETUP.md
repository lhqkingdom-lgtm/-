# Memory Lab 交接安装与恢复说明

这份文档用于把当前 Memory Lab 实验环境带到另一台机器继续跑。

## 1. 拉取代码

```powershell
git clone https://github.com/lhqkingdom-lgtm/-.git Codepulse-Memory-Lab
cd Codepulse-Memory-Lab
```

如果已经有目录：

```powershell
cd Codepulse-Memory-Lab
git pull origin main
```

## 2. 准备 Python

项目通过 `memory-lab.ps1` 启动。脚本会优先使用：

```text
%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
```

如果目标机器没有这个 Python，可以安装 Python 3.11+，然后设置：

```powershell
$env:MEMORY_LAB_PYTHON = "C:\Path\To\python.exe"
```

## 3. 配置 DeepSeek Key

仓库里不会提交真实 Key，需要在本机创建：

```text
memory-lab.secrets.local.json
```

内容格式：

```json
{
  "schema_version": "memory-lab-secrets-v1",
  "deepseek_api_key": "在这里填写你自己的 DeepSeek Key"
}
```

`memory-lab.config.json` 默认读取这个文件：

```json
{
  "schema_version": "memory-lab-config-v1",
  "llm": {
    "provider": "deepseek",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-flash",
    "allow_remote": true,
    "timeout_seconds": 120,
    "max_attempts": 3,
    "credential_file": "memory-lab.secrets.local.json"
  }
}
```

注意：

- 不要把 `memory-lab.secrets.local.json` 上传 GitHub。
- 如果 Key 曾经在聊天中暴露过，应先撤销并换新。

## 4. 恢复 CP SQLite 原始会话

当前原始 CP SQLite 已打包为：

```text
exports/cp-sqlite-raw-20260911-181147.zip
```

这个压缩包没有上传 GitHub，需要单独拷贝到目标机器。

解压后包含：

```text
opencode-dev.db
opencode-dev.db-wal
opencode-dev.db-shm
```

目标机器推荐放到：

```text
C:\Users\<用户名>\.local\share\codepulse\
```

也可以放到任意目录，然后运行命令时用 `-SourceDatabase` 指定：

```powershell
.\memory-lab.ps1 evidence-pools `
  -SourceDatabase "D:\somewhere\opencode-dev.db" `
  -LabDatabase "data\fanguard_domain_auto_rerun_20260911.db" `
  -SettleMilliseconds 0 `
  -Limit 10 `
  -EvidenceLayer all
```

注意：

- 这是原始会话库，可能包含完整对话、本机路径、工具输出，不要公开上传。
- 如果 CP 正在运行，数据库可能被占用；复制时建议先关 CP，或者复制 `.db/.db-wal/.db-shm` 三个文件形成快照。

## 5. Memory Lab 实验 DB

`data/` 默认不进 Git。

如果需要复现当前处理状态，还需要单独拷贝：

```text
data\fanguard_domain_auto_rerun_20260911.db
```

如果没有这个实验 DB，也可以从 CP SQLite 重新扫描生成。

## 6. PostgreSQL 事实源

当前重构后的三层证据池和长期记忆表保存在 PostgreSQL：

```text
host     = localhost
port     = 55512
database = codepulse_memory_lab
user     = root
password = 见本机 pg-memory-lab.local.json
```

本机数据目录：

```text
D:\codepulse\Codepulse-Memory-Lab\pgdata
```

历史上该实例使用 `55432`，但当前 Windows 将 `55432` 放入 TCP 排除端口段 `55412–55511`，所以本机改用同一套 `pgdata` 启动到 `55512`。

启动命令：

```powershell
& D:\PostgreSQL\17\bin\pg_ctl.exe start `
  -D D:\codepulse\Codepulse-Memory-Lab\pgdata `
  -o "-p 55512 -h localhost" `
  -l D:\codepulse\Codepulse-Memory-Lab\pgdata\memory-lab-postgres-55512.log
```

确认命令：

```powershell
.\memory-lab.ps1 pg-quality-report -Limit 5
```

当前已确认该实例包含：

```text
turns = 9
evidence_items = 386
memory_topics = 1
memories = 1
```

## 7. 常用命令

### 查看 turn 阶段粗识别

```powershell
.\memory-lab.ps1 turn-phases `
  -SourceDatabase "C:\Users\<用户名>\.local\share\codepulse\opencode-dev.db" `
  -LabDatabase "data\fanguard_domain_auto_rerun_20260911.db" `
  -SettleMilliseconds 0 `
  -Limit 20
```

### 只看 core 层

```powershell
.\memory-lab.ps1 evidence-pools `
  -SourceDatabase "C:\Users\<用户名>\.local\share\codepulse\opencode-dev.db" `
  -LabDatabase "data\fanguard_domain_auto_rerun_20260911.db" `
  -SettleMilliseconds 0 `
  -Limit 10 `
  -EvidenceLayer core
```

### 只看 candidate 层

```powershell
.\memory-lab.ps1 evidence-pools `
  -SourceDatabase "C:\Users\<用户名>\.local\share\codepulse\opencode-dev.db" `
  -LabDatabase "data\fanguard_domain_auto_rerun_20260911.db" `
  -SettleMilliseconds 0 `
  -Limit 10 `
  -EvidenceLayer candidate
```

### 只看 pointer 层

```powershell
.\memory-lab.ps1 evidence-pools `
  -SourceDatabase "C:\Users\<用户名>\.local\share\codepulse\opencode-dev.db" `
  -LabDatabase "data\fanguard_domain_auto_rerun_20260911.db" `
  -SettleMilliseconds 0 `
  -Limit 10 `
  -EvidenceLayer pointer
```

### 看 DeepSeek 阶段改判

```powershell
.\memory-lab.ps1 evidence-pools `
  -SourceDatabase "C:\Users\<用户名>\.local\share\codepulse\opencode-dev.db" `
  -LabDatabase "data\fanguard_domain_auto_rerun_20260911.db" `
  -SettleMilliseconds 0 `
  -Limit 5 `
  -EvidenceLayer router `
  -ModelRouter
```

## 7. 当前功能状态

已经实现：

- 从 CP SQLite 只读扫描会话；
- 按一轮任务切 turn；
- 阶段粗识别；
- 三层证据池：
  - `core`
  - `candidate`
  - `pointer`
- DeepSeek 阶段改判；
- 真实 FanGuard/MJ60B 会话验证；
- 边界处理：
  - 普通 `read/edit/write/glob/grep` 不进 core；
  - 需求选项进入 candidate；
  - 用户现象、助手根因、用户确认、真实编译结果进入 core。

暂未完成：

- PostgreSQL 事实源；
- LangGraph 工作流；
- 模型补证据请求；
- pointer 回查补证据；
- Writer / Reviewer / Committer 重构；
- PostgreSQL 到 Markdown 的正式投影。

详细重构方案见：

```text
LANGGRAPH_POSTGRES_REFACTOR_PLAN.md
```

## 8. 推荐继续顺序

1. 先在目标机器恢复 CP SQLite。
2. 跑 `turn-phases` 确认能读到真实 turn。
3. 分别跑 `EvidenceLayer core/candidate/pointer`，确认三层输出正常。
4. 配好 DeepSeek 后跑 `EvidenceLayer router -ModelRouter`。
5. 再开始 PostgreSQL + LangGraph 重构。
