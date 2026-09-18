# GLM5.1 公司密钥接入说明

当前 Memory Lab 的模型客户端使用 OpenAI-compatible Chat Completions 协议，因此 GLM5.1 只需要配置 `base_url`、`model` 和密钥来源即可。

## 推荐方式：环境变量

先复制示例配置：

```powershell
Copy-Item .\memory-lab.glm.config.example.json .\memory-lab.glm.config.local.json
```

然后在当前 PowerShell 里设置公司密钥：

```powershell
$env:MEMORY_LAB_GLM_API_KEY = "你的公司 GLM 密钥"
```

测试连通性：

```powershell
.\memory-lab.ps1 llm-smoke-test -ConfigPath .\memory-lab.glm.config.local.json
```

成功时会返回类似：

```json
{
  "ok": true,
  "status": "complete",
  "provider": "glm",
  "model": "glm-5.1",
  "result": {
    "ok": true,
    "message": "pong"
  }
}
```

## 备选方式：本地密钥文件

如果不想每次设置环境变量，可以把配置改成：

```json
{
  "schema_version": "memory-lab-config-v1",
  "llm": {
    "provider": "glm",
    "base_url": "https://open.bigmodel.cn/api/paas/v4",
    "model": "glm-5.1",
    "allow_remote": true,
    "timeout_seconds": 120,
    "max_attempts": 3,
    "credential_file": "memory-lab.glm.secrets.local.json"
  }
}
```

再新建 `memory-lab.glm.secrets.local.json`：

```json
{
  "schema_version": "memory-lab-secrets-v1",
  "glm_api_key": "你的公司 GLM 密钥"
}
```

这些 `.local.json` 和 `memory-lab.secrets*.json` 文件已被 `.gitignore` 忽略，不会被正常提交。

## 接入后如何用于记忆流程

连通性测试通过后，运行 ReAct 记忆整理时指定 GLM 配置即可：

```powershell
.\memory-lab.ps1 pg-memory-react-run -TurnId <turn_id> -ConfigPath .\memory-lab.glm.config.local.json -MemoryLimit 1 -MaxIterations 3 -DryRun
```

先用 `-DryRun` 看模型调用链是否合理，再去掉 `-DryRun` 真实提交。

## 安全边界

- 不要把真实密钥写进 `memory-lab.config.json`。
- 不要把真实密钥发到聊天里。
- smoke test 输出只显示密钥来源，不输出密钥值。
- 配置里的 `api_key` 明文字段会被程序拒绝。
