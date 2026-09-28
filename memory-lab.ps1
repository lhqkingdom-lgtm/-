param(
    [Parameter(Position = 0, Mandatory = $true)]
    [ValidateSet('baseline', 'ingest', 'historical-backfill', 'prepare', 'coverage', 'admit', 'turns', 'episodes', 'turn-phases', 'evidence-pools', 'conflict-scenarios-report', 'pg-ingest-turn', 'pg-show-turn', 'pg-quality-report', 'pg-writer-context', 'pg-get-turn-evidence', 'pg-migrate-memory-versioning', 'pg-current-turn-evidence-summary', 'pg-list-memory-topics', 'pg-list-topic-memories', 'pg-read-memory-history', 'pg-list-memory-conflicts', 'pg-expand-pointer', 'pg-read-memory-detail', 'pg-trace-memory-relations', 'pg-read-topic-detail', 'pg-search-related-memories', 'pg-export-markdown', 'pg-writer-dry-run', 'pg-commit-plan', 'pg-writer-commit', 'pg-validate-memory-changes', 'pg-commit-memory-changes', 'pg-memory-agent-run', 'pg-memory-react-run', 'pg-memory-agent-batch', 'process-new-turns', 'poll-new-turns', 'trace-report', 'eval-memory', 'llm-smoke-test', 'review', 'consolidate', 'shadow', 'report', 'catalog', 'recall', 'markdown-preview', 'markdown-recall', 'auto-memory', 'formal-publish', 'change-report', 'formal-recall', 'cycle', 'session-memory', 'watch', 'engineering-candidates', 'topics', 'rollback-memory', 'soak-start', 'soak-status')]
    [string]$Command,

    [string]$SourceDatabase = "$env:USERPROFILE\.local\share\codepulse\opencode-dev.db",

    [string]$LabDatabase = "$PSScriptRoot\data\memory_lab.db",

    [ValidateRange(0, 60000)]
    [int]$SettleMilliseconds = 1000,

    [string]$ProposalId,

    [ValidateSet('approve', 'reject')]
    [string]$ReviewOutcome,

    [string]$Reviewer,

    [string]$Note,

    [string]$ConfigPath = "$PSScriptRoot\memory-lab.config.json",

    [string]$PgConfigPath = "$PSScriptRoot\pg-memory-lab.local.json",

    [string]$GoldPath = "$PSScriptRoot\gold\bootstrap_v1.json",

    [string]$ReportDirectory = "$PSScriptRoot\reports",

    [string]$AgentLogDirectory,

    [string]$MemoryDirectory = "$PSScriptRoot\preview-memory",

    [string]$FormalMemoryRoot = "$PSScriptRoot\memory",

    [string]$ShadowRunId,

    [string]$RunId,

    [string]$TurnId,

    [string[]]$TurnIds,

    [string[]]$SessionIds,

    [string]$RawPointerId,

    [ValidateRange(1, 20000)]
    [int]$MaxChars = 4000,

    [ValidateRange(1, 50)]
    [int]$MaxItemsPerLayer = 12,

    [ValidateRange(1, 500)]
    [int]$MemoryLimit = 80,

    [ValidateRange(1, 10)]
    [int]$MaxIterations = 3,

    [string]$Query,

    [string]$MemoryType,

    [string]$ProjectDirectory,

    [string]$Domain,

    [string]$ProjectKey,

    [string[]]$Condition,

    [ValidateRange(1, 50)]
    [int]$Limit = 5,

    [ValidateRange(1, 32)]
    [int]$BatchSize = 32,

    [ValidateRange(1, 5)]
    [int]$TopicLimit = 5,

    [ValidateRange(1, 3600)]
    [double]$IntervalSeconds = 15,

    [int]$MaxCycles,

    [string]$MemoryId,

    [string]$TopicId,

    [string]$ConflictStatus,

    [string]$TargetRevisionId,

    [string]$Reason,

    [ValidateSet('backward', 'forward', 'both')]
    [string]$RelationDirection = 'both',

    [string]$Name,

    [string]$PlanPath,

    [string]$EvalSetPath = "$PSScriptRoot\evalsets\memory_eval_v1",

    [string]$RunName = 'single-batch',

    [ValidateSet('plan', 'single-batch', 'single-isolated', 'scenario-sequential', 'review-existing')]
    [string]$EvalMode = 'single-batch',

    [string]$ExistingReportDirectory,

    [switch]$ModelRouter,

    [ValidateSet('all', 'core', 'candidate', 'pointer', 'router')]
    [string]$EvidenceLayer = 'all',

    [ValidateSet('core', 'candidate', 'pointer')]
    [string[]]$Layer,

    [switch]$IncludeInactive,

    [switch]$IncludeTestData,

    [switch]$IncludeResolved,

    [switch]$Refresh,

    [switch]$Reprocess,

    [switch]$AutoTraceReport,

    [switch]$DryRun,

    [switch]$DryRunPlan
)

$BundledPythonExecutable = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$PythonExecutable = $env:MEMORY_LAB_PYTHON
if (-not $PythonExecutable) {
    if (Test-Path -LiteralPath $BundledPythonExecutable -PathType Leaf) {
        $PythonExecutable = $BundledPythonExecutable
    }
    else {
        $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($PythonCommand) {
            $PythonExecutable = $PythonCommand.Source
        }
    }
}
$SourceDirectory = "$PSScriptRoot\src"
$MemoryLabScriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $MemoryLabScriptRoot) {
    $MemoryLabScriptRoot = (Get-Location).Path
}

foreach ($PathVariableName in @('SourceDatabase', 'LabDatabase', 'ConfigPath', 'PgConfigPath', 'GoldPath', 'ReportDirectory', 'AgentLogDirectory', 'MemoryDirectory', 'FormalMemoryRoot')) {
    $PathValue = Get-Variable -Name $PathVariableName -ValueOnly -ErrorAction SilentlyContinue
    if ($PathValue -is [string] -and $PathValue.Length -ge 2 -and $PathValue.StartsWith('"') -and $PathValue.EndsWith('"')) {
        Set-Variable -Name $PathVariableName -Value $PathValue.Trim('"')
    }
}

if ($LabDatabase -eq '\data\memory_lab.db' -or $LabDatabase -eq '\data\\memory_lab.db' -or [string]::IsNullOrWhiteSpace($LabDatabase)) {
    $LabDatabase = Join-Path $MemoryLabScriptRoot 'data\memory_lab.db'
}
if ($ConfigPath -eq '\memory-lab.config.json' -or [string]::IsNullOrWhiteSpace($ConfigPath)) {
    $ConfigPath = Join-Path $MemoryLabScriptRoot 'memory-lab.config.json'
}
if ($PgConfigPath -eq '\pg-memory-lab.local.json' -or [string]::IsNullOrWhiteSpace($PgConfigPath)) {
    $PgConfigPath = Join-Path $MemoryLabScriptRoot 'pg-memory-lab.local.json'
}
if ($ReportDirectory -eq '\reports' -or [string]::IsNullOrWhiteSpace($ReportDirectory)) {
    $ReportDirectory = Join-Path $MemoryLabScriptRoot 'reports'
}
if ($MemoryDirectory -eq '\preview-memory' -or [string]::IsNullOrWhiteSpace($MemoryDirectory)) {
    $MemoryDirectory = Join-Path $MemoryLabScriptRoot 'preview-memory'
}
if ($FormalMemoryRoot -eq '\memory' -or [string]::IsNullOrWhiteSpace($FormalMemoryRoot)) {
    $FormalMemoryRoot = Join-Path $MemoryLabScriptRoot 'memory'
}
$SourceDirectory = Join-Path $MemoryLabScriptRoot 'src'

if (-not $PythonExecutable -or -not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw 'Python 3.11+ was not found. Set MEMORY_LAB_PYTHON to an absolute python.exe path.'
}

function Get-MemoryLabProperty {
    param(
        [object]$Object,
        [string]$Name
    )
    if ($null -eq $Object) {
        return $null
    }
    $Property = $Object.PSObject.Properties[$Name]
    if ($Property) {
        return $Property.Value
    }
    return $null
}

function Get-MemoryLabShortText {
    param(
        [object]$Value,
        [int]$Limit = 180
    )
    if ($null -eq $Value) {
        return ''
    }
    $Text = ([string]$Value) -replace '\s+', ' '
    $Text = $Text.Trim()
    if ($Text.Length -le $Limit) {
        return $Text
    }
    if ($Limit -le 1) {
        return '.'
    }
    if ($Limit -le 3) {
        return '.' * $Limit
    }
    return $Text.Substring(0, $Limit - 3) + '...'
}

function ConvertTo-MemoryLabCompactJson {
    param(
        [object]$Value,
        [int]$Limit = 180
    )
    if ($null -eq $Value) {
        return ''
    }
    try {
        $Json = $Value | ConvertTo-Json -Compress -Depth 12
        return Get-MemoryLabShortText $Json $Limit
    }
    catch {
        return Get-MemoryLabShortText $Value $Limit
    }
}

function Get-MemoryLabToolResultSummary {
    param([object]$Result)
    if ($null -eq $Result) {
        return ''
    }
    $Status = Get-MemoryLabProperty $Result 'status'
    $Memory = Get-MemoryLabProperty $Result 'memory'
    if ($Memory) {
        $Topic = Get-MemoryLabProperty $Memory 'topic'
        $TopicTitle = if ($Topic) { Get-MemoryLabProperty $Topic 'title' } else { $null }
        $Summary = Get-MemoryLabProperty $Memory 'summary'
        return Get-MemoryLabShortText ("status={0}; memory={1}; topic={2}; summary={3}" -f `
            $Status, (Get-MemoryLabProperty $Memory 'memory_id'), $TopicTitle, $Summary) 260
    }
    $TopicObject = Get-MemoryLabProperty $Result 'topic'
    if (-not $TopicObject) {
        $Detail = Get-MemoryLabProperty $Result 'detail'
        if ($Detail) {
            $TopicObject = Get-MemoryLabProperty $Detail 'topic'
        }
    }
    if ($TopicObject) {
        return Get-MemoryLabShortText ("status={0}; topic={1}; title={2}; summary={3}" -f `
            $Status, (Get-MemoryLabProperty $TopicObject 'topic_id'), (Get-MemoryLabProperty $TopicObject 'title'), (Get-MemoryLabProperty $TopicObject 'summary')) 260
    }
    foreach ($ListName in @('memories', 'conflicts', 'revisions', 'evidence', 'items', 'rows')) {
        $ListValue = Get-MemoryLabProperty $Result $ListName
        if ($ListValue -is [array]) {
            $Samples = @($ListValue | Select-Object -First 3)
            return Get-MemoryLabShortText ("status={0}; {1}_count={2}; sample={3}" -f `
                $Status, $ListName, @($ListValue).Count, (ConvertTo-MemoryLabCompactJson $Samples 220)) 360
        }
    }
    return ConvertTo-MemoryLabCompactJson $Result 360
}

function Write-MemoryLabReactSummary {
    param(
        [object]$Turn,
        [string]$LogDirectory
    )
    $TurnId = Get-MemoryLabProperty $Turn 'turn_id'
    $RunId = Get-MemoryLabProperty $Turn 'agent_run_id'
    $ActionCounts = Get-MemoryLabProperty $Turn 'action_counts'
    $ActionSummary = 'none'
    if ($ActionCounts) {
        $Pairs = @()
        foreach ($Property in $ActionCounts.PSObject.Properties) {
            $Pairs += "$($Property.Name)=$($Property.Value)"
        }
        if ($Pairs.Count -gt 0) {
            $ActionSummary = ($Pairs | Sort-Object) -join ', '
        }
    }
    Write-Host ("  [本轮] {0} 状态={1} 记忆Agent={2} 动作={3}" -f `
        (Get-MemoryLabShortText $TurnId 22), (Get-MemoryLabProperty $Turn 'status'), (Get-MemoryLabProperty $Turn 'react_status'), $ActionSummary)
    if (-not $RunId) {
        Write-Host "  [记忆Agent] 没有 agent_run_id，无法读取详细日志。"
        return
    }
    if (-not $LogDirectory) {
        Write-Host ("  [记忆Agent] run_id={0}；未设置 -AgentLogDirectory，无法读取详细日志。" -f $RunId)
        return
    }
    $LogPath = Join-Path $LogDirectory "$RunId.json"
    if (-not (Test-Path -LiteralPath $LogPath -PathType Leaf)) {
        Write-Host ("  [记忆Agent] run_id={0}；详细日志不存在：{1}" -f $RunId, $LogPath)
        return
    }
    try {
        $Log = Get-Content -LiteralPath $LogPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        Write-Host ("  [记忆Agent] run_id={0}；详细日志解析失败：{1}" -f $RunId, $_.Exception.Message)
        return
    }
    Write-Host ("  [记忆Agent] run_id={0} 模式={1} 状态={2}" -f $RunId, (Get-MemoryLabProperty $Log 'mode'), (Get-MemoryLabProperty $Log 'status'))
    foreach ($Step in @((Get-MemoryLabProperty $Log 'steps'))) {
        Write-Host ("  [步骤] {0}: {1}" -f (Get-MemoryLabProperty $Step 'step'), (Get-MemoryLabShortText (Get-MemoryLabProperty $Step 'summary') 160))
    }
    foreach ($Iteration in @((Get-MemoryLabProperty $Log 'iterations'))) {
        $Raw = Get-MemoryLabProperty $Iteration 'raw'
        $ToolRequests = @()
        if ($Raw) {
            $ToolRequests = @((Get-MemoryLabProperty $Raw 'tool_requests'))
        }
        $ToolNames = @()
        foreach ($Request in $ToolRequests) {
            $ToolNames += (Get-MemoryLabProperty $Request 'tool')
        }
        $ToolText = if ($ToolNames.Count -gt 0) { ($ToolNames -join ', ') } else { 'none' }
        $Reason = if ($Raw) { Get-MemoryLabProperty $Raw 'reason' } else { $null }
        Write-Host ("  [判断第 {0} 轮] 决策={1} 请求工具={2}；原因={3}" -f `
            (Get-MemoryLabProperty $Iteration 'iteration'), (Get-MemoryLabProperty $Iteration 'decision'), $ToolText, (Get-MemoryLabShortText $Reason 220))
    }
    foreach ($Observation in @((Get-MemoryLabProperty $Log 'observations'))) {
        $Request = Get-MemoryLabProperty $Observation 'request'
        $Result = Get-MemoryLabProperty $Observation 'result'
        $Tool = Get-MemoryLabProperty $Request 'tool'
        $Reason = Get-MemoryLabProperty $Request 'reason'
        $Parameters = @{}
        foreach ($Name in @('memory_id', 'topic_id', 'turn_id', 'raw_pointer_id', 'query', 'limit', 'layers', 'max_chars', 'preview_chars')) {
            $Value = Get-MemoryLabProperty $Request $Name
            if ($null -ne $Value -and "$Value" -ne '') {
                $Parameters[$Name] = $Value
            }
        }
        Write-Host ("  [工具] {0} 参数={1} -> 状态={2}；原因={3}；返回={4}" -f `
            $Tool, (ConvertTo-MemoryLabCompactJson $Parameters 120), (Get-MemoryLabProperty $Result 'status'), `
            (Get-MemoryLabShortText $Reason 180), (Get-MemoryLabToolResultSummary $Result))
    }
    $ResultObject = Get-MemoryLabProperty $Log 'result'
    $ChainState = if ($ResultObject) { Get-MemoryLabProperty $ResultObject 'chain_state' } else { $null }
    $ChainPersistence = if ($ResultObject) { Get-MemoryLabProperty $ResultObject 'chain_persistence' } else { $null }
    if ($ChainState) {
        $ChainId = Get-MemoryLabProperty $ChainState 'chain_id'
        if (-not $ChainId) { $ChainId = '新链（提交后生成）' }
        $RelatedTurnCount = @((Get-MemoryLabProperty $ChainState 'related_turn_ids')).Count
        $Missing = @((Get-MemoryLabProperty $ChainState 'missing_evidence'))
        $MissingText = if ($Missing.Count -gt 0) { $Missing -join '；' } else { '无' }
        $PersistStatus = if ($ChainPersistence) { Get-MemoryLabProperty $ChainPersistence 'status' } else { '未写入（DryRun 或非技术链）' }
        $PersistResult = if ($ChainPersistence) { Get-MemoryLabProperty $ChainPersistence 'result' } else { '' }
        Write-Host ("  [故障链] ID={0} 类型={1} 状态={2} 关联turn={3} 缺口={4}；保存状态={5} {6}" -f `
            $ChainId, (Get-MemoryLabProperty $ChainState 'chain_type'), (Get-MemoryLabProperty $ChainState 'state'), `
            $RelatedTurnCount, (Get-MemoryLabShortText $MissingText 220), $PersistStatus, $PersistResult)
    }
    $Plan = if ($ResultObject) { Get-MemoryLabProperty $ResultObject 'plan' } else { $null }
    if ($Plan) {
        Write-Host ("  [最终判断] {0}" -f (Get-MemoryLabShortText (Get-MemoryLabProperty $Plan 'reason') 260))
        foreach ($Change in @((Get-MemoryLabProperty $Plan 'changes'))) {
            $Memory = Get-MemoryLabProperty $Change 'memory'
            $Topic = Get-MemoryLabProperty $Change 'topic'
            $Target = Get-MemoryLabProperty $Change 'target_memory_id'
            if (-not $Target) {
                $Target = Get-MemoryLabProperty $Change 'target_topic_id'
            }
            if (-not $Target -and $Memory) {
                $Target = Get-MemoryLabProperty $Memory 'stable_key'
            }
            if (-not $Target -and $Topic) {
                $Target = Get-MemoryLabProperty $Topic 'title'
            }
            $Summary = if ($Memory) { Get-MemoryLabProperty $Memory 'summary' } elseif ($Topic) { Get-MemoryLabProperty $Topic 'summary' } else { Get-MemoryLabProperty $Change 'reason' }
            Write-Host ("  [记忆变化] 动作={0} 目标={1}；{2}" -f `
                (Get-MemoryLabProperty $Change 'action'), (Get-MemoryLabShortText $Target 80), (Get-MemoryLabShortText $Summary 260))
        }
    }
    Write-Host ("  [详细文件] {0}" -f $LogPath)
}

$PreviousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = $SourceDirectory
    if ($Command -in @('turns', 'episodes')) {
        & $PythonExecutable -m memory_lab --source-db $SourceDatabase --lab-db $LabDatabase $Command --settle-ms $SettleMilliseconds
    }
    elseif ($Command -eq 'turn-phases') {
        & $PythonExecutable -m memory_lab --source-db $SourceDatabase --lab-db $LabDatabase $Command --settle-ms $SettleMilliseconds --limit $Limit
    }
    elseif ($Command -eq 'evidence-pools') {
        $EvidenceArguments = @(
            '-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase,
            $Command, '--settle-ms', $SettleMilliseconds, '--limit', $Limit,
            '--config', $ConfigPath, '--layer', $EvidenceLayer
        )
        if ($ModelRouter) {
            $EvidenceArguments += '--model-router'
        }
        & $PythonExecutable @EvidenceArguments
    }
    elseif ($Command -eq 'historical-backfill') {
        if (-not $SessionIds -or $SessionIds.Count -eq 0) {
            throw 'historical-backfill requires -SessionIds'
        }
        $Arguments = @('-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase, $Command)
        foreach ($SessionId in $SessionIds) {
            $Arguments += @('--session-id', $SessionId)
        }
        & $PythonExecutable @Arguments
    }
    elseif ($Command -eq 'pg-ingest-turn') {
        $PgArguments = @(
            '-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase,
            $Command, '--settle-ms', $SettleMilliseconds, '--pg-config', $PgConfigPath
        )
        if ($TurnId) {
            $PgArguments += @('--turn-id', $TurnId)
        }
        & $PythonExecutable @PgArguments
    }
    elseif ($Command -eq 'pg-show-turn') {
        if (-not $TurnId) {
            throw 'pg-show-turn requires -TurnId'
        }
        & $PythonExecutable -m memory_lab --source-db $SourceDatabase --lab-db $LabDatabase `
            $Command --pg-config $PgConfigPath --turn-id $TurnId
    }
    elseif ($Command -eq 'pg-quality-report') {
        & $PythonExecutable -m memory_lab --source-db $SourceDatabase --lab-db $LabDatabase `
            $Command --pg-config $PgConfigPath --limit $Limit
    }
    elseif ($Command -eq 'conflict-scenarios-report') {
        & $PythonExecutable -m memory_lab --source-db $SourceDatabase --lab-db $LabDatabase `
            $Command --output-dir (Join-Path $ReportDirectory 'conflict-scenarios')
    }
    elseif ($Command -eq 'pg-writer-context') {
        if (-not $TurnId) {
            throw 'pg-writer-context requires -TurnId'
        }
        & $PythonExecutable -m memory_lab --source-db $SourceDatabase --lab-db $LabDatabase `
            $Command --pg-config $PgConfigPath --turn-id $TurnId --memory-limit $MemoryLimit
    }
    elseif ($Command -eq 'pg-get-turn-evidence') {
        if (-not $TurnId) {
            throw 'pg-get-turn-evidence requires -TurnId'
        }
        $Arguments = @(
            '-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase,
            $Command, '--pg-config', $PgConfigPath, '--turn-id', $TurnId, '--max-chars', $MaxChars
        )
        foreach ($Item in ($Layer | Where-Object { $_ })) {
            $Arguments += @('--layer', $Item)
        }
        & $PythonExecutable @Arguments
    }
    elseif ($Command -eq 'pg-migrate-memory-versioning') {
        & $PythonExecutable -m memory_lab --source-db $SourceDatabase --lab-db $LabDatabase `
            $Command --pg-config $PgConfigPath
    }
    elseif ($Command -eq 'pg-current-turn-evidence-summary') {
        if (-not $TurnId) {
            throw 'pg-current-turn-evidence-summary requires -TurnId'
        }
        & $PythonExecutable -m memory_lab --source-db $SourceDatabase --lab-db $LabDatabase `
            $Command --pg-config $PgConfigPath --turn-id $TurnId `
            --max-items-per-layer $MaxItemsPerLayer --max-chars $MaxChars
    }
    elseif ($Command -eq 'pg-list-memory-topics') {
        $Arguments = @(
            '-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase,
            $Command, '--pg-config', $PgConfigPath, '--limit', $Limit
        )
        if ($Domain) {
            $Arguments += @('--domain', $Domain)
        }
        if ($ProjectKey) {
            $Arguments += @('--project-key', $ProjectKey)
        }
        if ($IncludeInactive) {
            $Arguments += '--include-inactive'
        }
        if ($IncludeTestData) {
            $Arguments += '--include-test-data'
        }
        & $PythonExecutable @Arguments
    }
    elseif ($Command -eq 'pg-expand-pointer') {
        if (-not $RawPointerId) {
            throw 'pg-expand-pointer requires -RawPointerId'
        }
        & $PythonExecutable -m memory_lab --source-db $SourceDatabase --lab-db $LabDatabase `
            $Command --pg-config $PgConfigPath --raw-pointer-id $RawPointerId --max-chars $MaxChars
    }
    elseif ($Command -eq 'pg-read-memory-detail') {
        if (-not $MemoryId) {
            throw 'pg-read-memory-detail requires -MemoryId'
        }
        & $PythonExecutable -m memory_lab --source-db $SourceDatabase --lab-db $LabDatabase `
            $Command --pg-config $PgConfigPath --memory-id $MemoryId
    }
    elseif ($Command -eq 'pg-list-topic-memories') {
        if (-not $TopicId) {
            throw 'pg-list-topic-memories requires -TopicId'
        }
        $Arguments = @(
            '-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase,
            $Command, '--pg-config', $PgConfigPath, '--topic-id', $TopicId, '--limit', $Limit
        )
        if ($IncludeInactive) {
            $Arguments += '--include-inactive'
        }
        if ($IncludeTestData) {
            $Arguments += '--include-test-data'
        }
        & $PythonExecutable @Arguments
    }
    elseif ($Command -eq 'pg-read-memory-history') {
        if (-not $MemoryId) {
            throw 'pg-read-memory-history requires -MemoryId'
        }
        & $PythonExecutable -m memory_lab --source-db $SourceDatabase --lab-db $LabDatabase `
            $Command --pg-config $PgConfigPath --memory-id $MemoryId --limit $Limit
    }
    elseif ($Command -eq 'pg-list-memory-conflicts') {
        $Arguments = @(
            '-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase,
            $Command, '--pg-config', $PgConfigPath, '--limit', $Limit
        )
        if ($MemoryId) {
            $Arguments += @('--memory-id', $MemoryId)
        }
        if ($TopicId) {
            $Arguments += @('--topic-id', $TopicId)
        }
        if ($IncludeTestData) {
            $Arguments += '--include-test-data'
        }
        if ($ConflictStatus) {
            $Arguments += @('--status', $ConflictStatus)
        }
        if ($IncludeResolved) {
            $Arguments += '--include-resolved'
        }
        & $PythonExecutable @Arguments
    }
    elseif ($Command -eq 'pg-trace-memory-relations') {
        if (-not $MemoryId) {
            throw 'pg-trace-memory-relations requires -MemoryId'
        }
        $Arguments = @(
            '-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase,
            $Command, '--pg-config', $PgConfigPath, '--memory-id', $MemoryId,
            '--direction', $RelationDirection, '--max-depth', $MaxIterations
        )
        if ($IncludeTestData) {
            $Arguments += '--include-test-data'
        }
        & $PythonExecutable @Arguments
    }
    elseif ($Command -eq 'pg-read-topic-detail') {
        if (-not $TopicId) {
            throw 'pg-read-topic-detail requires -TopicId'
        }
        $Arguments = @(
            '-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase,
            $Command, '--pg-config', $PgConfigPath, '--topic-id', $TopicId, '--max-chars', $MaxChars
        )
        if ($IncludeInactive) {
            $Arguments += '--include-inactive'
        }
        if ($IncludeTestData) {
            $Arguments += '--include-test-data'
        }
        & $PythonExecutable @Arguments
    }
    elseif ($Command -eq 'pg-search-related-memories') {
        if (-not $Query) {
            throw 'pg-search-related-memories requires -Query'
        }
        $Arguments = @(
            '-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase,
            $Command, '--pg-config', $PgConfigPath, '--query', $Query, '--limit', $Limit
        )
        if ($MemoryType) {
            $Arguments += @('--memory-type', $MemoryType)
        }
        if ($Domain) {
            $Arguments += @('--domain', $Domain)
        }
        if ($ProjectKey) {
            $Arguments += @('--project-key', $ProjectKey)
        }
        if ($IncludeInactive) {
            $Arguments += '--include-inactive'
        }
        if ($IncludeTestData) {
            $Arguments += '--include-test-data'
        }
        & $PythonExecutable @Arguments
    }
    elseif ($Command -eq 'pg-writer-dry-run') {
        if (-not $TurnId) {
            throw 'pg-writer-dry-run requires -TurnId'
        }
        & $PythonExecutable -m memory_lab --source-db $SourceDatabase --lab-db $LabDatabase `
            $Command --pg-config $PgConfigPath --config $ConfigPath `
            --turn-id $TurnId --memory-limit $MemoryLimit
    }
    elseif ($Command -eq 'llm-smoke-test') {
        & $PythonExecutable -m memory_lab $Command --config $ConfigPath
    }
    elseif ($Command -eq 'pg-commit-plan') {
        if (-not $PlanPath) {
            throw 'pg-commit-plan requires -PlanPath'
        }
        & $PythonExecutable -m memory_lab --source-db $SourceDatabase --lab-db $LabDatabase `
            $Command --pg-config $PgConfigPath --plan-path $PlanPath
    }
    elseif ($Command -eq 'pg-writer-commit') {
        if (-not $TurnId) {
            throw 'pg-writer-commit requires -TurnId'
        }
        & $PythonExecutable -m memory_lab --source-db $SourceDatabase --lab-db $LabDatabase `
            $Command --pg-config $PgConfigPath --config $ConfigPath `
            --turn-id $TurnId --memory-limit $MemoryLimit
    }
    elseif ($Command -eq 'pg-validate-memory-changes') {
        if (-not $PlanPath) {
            throw 'pg-validate-memory-changes requires -PlanPath'
        }
        & $PythonExecutable -m memory_lab --source-db $SourceDatabase --lab-db $LabDatabase `
            $Command --pg-config $PgConfigPath --plan-path $PlanPath
    }
    elseif ($Command -eq 'pg-commit-memory-changes') {
        if (-not $PlanPath) {
            throw 'pg-commit-memory-changes requires -PlanPath'
        }
        & $PythonExecutable -m memory_lab --source-db $SourceDatabase --lab-db $LabDatabase `
            $Command --pg-config $PgConfigPath --plan-path $PlanPath
    }
    elseif ($Command -eq 'pg-memory-agent-run') {
        if (-not $TurnId) {
            throw 'pg-memory-agent-run requires -TurnId'
        }
        $Arguments = @(
            '-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase,
            $Command, '--pg-config', $PgConfigPath, '--config', $ConfigPath,
            '--turn-id', $TurnId, '--memory-limit', $MemoryLimit
        )
        if ($PSBoundParameters.ContainsKey('TopicLimit')) {
            $Arguments += @('--topic-limit', $TopicLimit)
        }
        if ($DryRun) {
            $Arguments += '--dry-run'
        }
        if ($AgentLogDirectory) {
            $Arguments += @('--log-dir', $AgentLogDirectory)
        }
        & $PythonExecutable @Arguments
    }
    elseif ($Command -eq 'pg-memory-react-run') {
        if (-not $TurnId) {
            throw 'pg-memory-react-run requires -TurnId'
        }
        $Arguments = @(
            '-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase,
            $Command, '--pg-config', $PgConfigPath, '--config', $ConfigPath,
            '--turn-id', $TurnId, '--memory-limit', $MemoryLimit, '--max-iterations', $MaxIterations
        )
        if ($PSBoundParameters.ContainsKey('TopicLimit')) {
            $Arguments += @('--topic-limit', $TopicLimit)
        }
        if ($DryRun) {
            $Arguments += '--dry-run'
        }
        if ($AgentLogDirectory) {
            $Arguments += @('--log-dir', $AgentLogDirectory)
        }
        & $PythonExecutable @Arguments
    }
    elseif ($Command -eq 'pg-memory-agent-batch') {
        $Arguments = @(
            '-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase,
            $Command, '--pg-config', $PgConfigPath, '--config', $ConfigPath,
            '--limit', $Limit, '--memory-limit', $MemoryLimit
        )
        if ($PSBoundParameters.ContainsKey('TopicLimit')) {
            $Arguments += @('--topic-limit', $TopicLimit)
        }
        foreach ($Item in ($TurnIds | Where-Object { $_ })) {
            $Arguments += @('--turn-id', $Item)
        }
        if ($TurnId) {
            $Arguments += @('--turn-id', $TurnId)
        }
        if ($DryRun) {
            $Arguments += '--dry-run'
        }
        if ($AgentLogDirectory) {
            $Arguments += @('--log-dir', $AgentLogDirectory)
        }
        & $PythonExecutable @Arguments
    }
    elseif ($Command -eq 'process-new-turns') {
        $Arguments = @(
            '-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase,
            $Command, '--pg-config', $PgConfigPath, '--config', $ConfigPath,
            '--limit', $Limit, '--settle-ms', $SettleMilliseconds,
            '--memory-limit', $MemoryLimit, '--max-iterations', $MaxIterations
        )
        if ($PSBoundParameters.ContainsKey('TopicLimit')) {
            $Arguments += @('--topic-limit', $TopicLimit)
        }
        if ($DryRun) {
            $Arguments += '--dry-run'
        }
        if ($Reprocess) {
            $Arguments += '--reprocess'
        }
        if ($AgentLogDirectory) {
            $Arguments += @('--log-dir', $AgentLogDirectory)
        }
        & $PythonExecutable @Arguments
    }
    elseif ($Command -eq 'poll-new-turns') {
        $Cycle = 0
        Write-Host "轮询服务已启动。CP数据库=$SourceDatabase 间隔=${IntervalSeconds}秒 每轮最多处理=$Limit DryRun=$DryRun MaxCycles=$MaxCycles"
        while ($true) {
            $Cycle += 1
            $StartedAt = Get-Date -Format o
            Write-Host "第 $Cycle 轮扫描开始：$StartedAt"
            $Arguments = @(
                '-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase,
                'process-new-turns', '--pg-config', $PgConfigPath, '--config', $ConfigPath,
                '--limit', $Limit, '--settle-ms', $SettleMilliseconds,
                '--memory-limit', $MemoryLimit, '--max-iterations', $MaxIterations
            )
            if ($PSBoundParameters.ContainsKey('TopicLimit')) {
                $Arguments += @('--topic-limit', $TopicLimit)
            }
            if ($DryRun) {
                $Arguments += '--dry-run'
            }
            if ($Reprocess) {
                $Arguments += '--reprocess'
            }
            if ($AgentLogDirectory) {
                $Arguments += @('--log-dir', $AgentLogDirectory)
            }
            $ProcessOutput = & $PythonExecutable @Arguments 2>&1
            $ProcessExitCode = $LASTEXITCODE
            if ($ProcessExitCode -ne 0) {
                Write-Warning "第 $Cycle 轮处理失败，退出码 $ProcessExitCode；服务会继续下一轮。"
                $ProcessOutput | ForEach-Object { Write-Warning $_ }
            }
            else {
                try {
                    $ProcessReport = ($ProcessOutput -join "`n") | ConvertFrom-Json
                    if ($AgentLogDirectory) {
                        New-Item -ItemType Directory -Force -Path $AgentLogDirectory | Out-Null
                        $CycleStamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
                        $CycleReportPath = Join-Path $AgentLogDirectory ("poll-cycle-{0}-{1}.json" -f $CycleStamp, $Cycle)
                        $LatestReportPath = Join-Path $AgentLogDirectory 'latest-process-new-turns.json'
                        ($ProcessOutput -join "`n") | Set-Content -LiteralPath $CycleReportPath -Encoding UTF8
                        ($ProcessOutput -join "`n") | Set-Content -LiteralPath $LatestReportPath -Encoding UTF8
                    }
                    $ActionCounts = @{}
                    foreach ($Turn in @($ProcessReport.turns)) {
                        if ($Turn.action_counts) {
                            foreach ($Property in $Turn.action_counts.PSObject.Properties) {
                                if (-not $ActionCounts.ContainsKey($Property.Name)) {
                                    $ActionCounts[$Property.Name] = 0
                                }
                                $ActionCounts[$Property.Name] += [int]$Property.Value
                            }
                        }
                    }
                    $ActionSummary = if ($ActionCounts.Count -gt 0) {
                        ($ActionCounts.GetEnumerator() | Sort-Object Name | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join ', '
                    }
                    else {
                        'none'
                    }
                    Write-Host ("第 {0} 轮结果：新turn={1} 已完成={2} 失败={3} 已跳过={4} 记忆动作={5}" -f `
                        $Cycle, $ProcessReport.pending_count, $ProcessReport.completed_count, $ProcessReport.failed_count, $ProcessReport.skipped_count, $ActionSummary)
                    if ([int]$ProcessReport.pending_count -gt 0) {
                        Write-Host ("第 {0} 轮详细流转：" -f $Cycle)
                        foreach ($Turn in @($ProcessReport.turns)) {
                            Write-MemoryLabReactSummary -Turn $Turn -LogDirectory $AgentLogDirectory
                        }
                        if ($AgentLogDirectory) {
                            Write-Host ("本轮详细日志目录：{0}" -f (Resolve-Path -LiteralPath $AgentLogDirectory).Path)
                        }
                    }
                }
                catch {
                    Write-Warning "第 $Cycle 轮已完成，但输出解析失败；下面打印原始输出。"
                    $ProcessOutput | ForEach-Object { Write-Host $_ }
                }
            }
            if ($AutoTraceReport) {
                $TraceOutputDirectory = Join-Path $ReportDirectory 'cp-memory-trace'
                $TraceArguments = @(
                    '-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase,
                    'trace-report', '--pg-config', $PgConfigPath,
                    '--output-dir', $TraceOutputDirectory,
                    '--limit', $Limit, '--max-chars', $MaxChars
                )
                if ($AgentLogDirectory) {
                    $TraceArguments += @('--log-dir', $AgentLogDirectory)
                }
                $TraceOutput = & $PythonExecutable @TraceArguments 2>&1
                $TraceExitCode = $LASTEXITCODE
                if ($TraceExitCode -ne 0) {
                    Write-Warning "trace-report failed with exit code $TraceExitCode; continuing next cycle."
                    $TraceOutput | ForEach-Object { Write-Warning $_ }
                }
                else {
                    try {
                        $TraceReport = ($TraceOutput -join "`n") | ConvertFrom-Json
                        $TraceHtml = if ($TraceReport.files.html) { $TraceReport.files.html } else { Join-Path $TraceOutputDirectory 'latest-trace.html' }
                        Write-Host "可视化报告已刷新：$TraceHtml"
                    }
                    catch {
                        Write-Host "可视化报告已刷新：$(Join-Path $TraceOutputDirectory 'latest-trace.html')"
                    }
                }
            }
            if ($MaxCycles -gt 0 -and $Cycle -ge $MaxCycles) {
                Write-Host "已达到 MaxCycles=$MaxCycles，轮询停止。"
                break
            }
            Start-Sleep -Seconds $IntervalSeconds
        }
    }
    elseif ($Command -eq 'trace-report') {
        $Arguments = @(
            '-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase,
            'trace-report', '--pg-config', $PgConfigPath,
            '--output-dir', (Join-Path $ReportDirectory 'cp-memory-trace'),
            '--limit', $Limit, '--max-chars', $MaxChars
        )
        if ($AgentLogDirectory) {
            $Arguments += @('--log-dir', $AgentLogDirectory)
        }
        & $PythonExecutable @Arguments
    }
    elseif ($Command -eq 'eval-memory') {
        $EvalArguments = @(
            '-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase,
            'eval-memory', '--eval-set', $EvalSetPath, '--pg-config', $PgConfigPath,
            '--config', $ConfigPath, '--report-dir', (Join-Path $ReportDirectory 'eval'),
            '--run-name', $RunName, '--mode', $EvalMode, '--limit', $Limit, '--max-iterations', $MaxIterations,
            '--memory-limit', $MemoryLimit, '--topic-limit', $TopicLimit
        )
        if ($DryRunPlan) {
            $EvalArguments += '--dry-run-plan'
        }
        if ($DryRun) {
            $EvalArguments += '--dry-run'
        }
        if ($ExistingReportDirectory) {
            $EvalArguments += @('--existing-report-dir', $ExistingReportDirectory)
        }
        & $PythonExecutable @EvalArguments
    }
    elseif ($Command -eq 'review') {
        if (-not $ProposalId -or -not $ReviewOutcome -or -not $Reviewer) {
            throw 'review requires -ProposalId, -ReviewOutcome, and -Reviewer'
        }
        $ReviewArguments = @(
            '-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase,
            $Command, '--proposal-id', $ProposalId, '--outcome', $ReviewOutcome,
            '--reviewer', $Reviewer
        )
        if ($Note) {
            $ReviewArguments += @('--note', $Note)
        }
        & $PythonExecutable @ReviewArguments
    }
    elseif ($Command -eq 'shadow') {
        $ShadowArguments = @(
            '-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase,
            $Command, '--config', $ConfigPath
        )
        if ($Refresh) {
            $ShadowArguments += '--refresh'
        }
        & $PythonExecutable @ShadowArguments
    }
    elseif ($Command -eq 'report') {
        $ReportArguments = @(
            '-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase,
            $Command, '--gold', $GoldPath, '--output-dir', $ReportDirectory
        )
        if ($ShadowRunId) {
            $ReportArguments += @('--shadow-run-id', $ShadowRunId)
        }
        & $PythonExecutable @ReportArguments
    }
    elseif ($Command -eq 'catalog') {
        $CatalogArguments = @(
            '-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase,
            $Command
        )
        if ($ShadowRunId) {
            $CatalogArguments += @('--shadow-run-id', $ShadowRunId)
        }
        & $PythonExecutable @CatalogArguments
    }
    elseif ($Command -eq 'recall') {
        if (-not $Query) {
            throw 'recall requires -Query'
        }
        $RecallArguments = @(
            '-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase,
            $Command, '--query', $Query, '--limit', $Limit
        )
        if ($ProjectDirectory) {
            $RecallArguments += @('--project-directory', $ProjectDirectory)
        }
        if ($Domain) {
            $RecallArguments += @('--domain', $Domain)
        }
        if ($ShadowRunId) {
            $RecallArguments += @('--shadow-run-id', $ShadowRunId)
        }
        & $PythonExecutable @RecallArguments
    }
    elseif ($Command -eq 'markdown-preview') {
        $MarkdownPreviewArguments = @(
            '-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase,
            $Command, '--memory-dir', $MemoryDirectory
        )
        if ($ShadowRunId) {
            $MarkdownPreviewArguments += @('--shadow-run-id', $ShadowRunId)
        }
        if ($DryRun) {
            $MarkdownPreviewArguments += '--dry-run'
        }
        & $PythonExecutable @MarkdownPreviewArguments
    }
    elseif ($Command -eq 'markdown-recall') {
        if (-not $Query) {
            throw 'markdown-recall requires -Query'
        }
        $MarkdownRecallArguments = @(
            '-m', 'memory_lab', $Command, '--memory-dir', $MemoryDirectory,
            '--query', $Query, '--limit', $Limit
        )
        if ($ProjectDirectory) {
            $MarkdownRecallArguments += @('--project-directory', $ProjectDirectory)
        }
        if ($Domain) {
            $MarkdownRecallArguments += @('--domain', $Domain)
        }
        & $PythonExecutable @MarkdownRecallArguments
    }
    elseif ($Command -eq 'auto-memory') {
        $AutoArguments = @(
            '-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase,
            $Command, '--config', $ConfigPath
        )
        if ($Refresh) {
            $AutoArguments += '--refresh'
        }
        & $PythonExecutable @AutoArguments
    }
    elseif ($Command -eq 'formal-publish') {
        & $PythonExecutable -m memory_lab --source-db $SourceDatabase --lab-db $LabDatabase `
            $Command --memory-root $FormalMemoryRoot
    }
    elseif ($Command -eq 'change-report') {
        if (-not $RunId) {
            throw 'change-report requires -RunId'
        }
        & $PythonExecutable -m memory_lab --source-db $SourceDatabase --lab-db $LabDatabase `
            $Command --run-id $RunId --output-dir $ReportDirectory `
            --memory-root $FormalMemoryRoot
    }
    elseif ($Command -eq 'formal-recall') {
        if (-not $Query) {
            throw 'formal-recall requires -Query'
        }
        $FormalRecallArguments = @(
            '-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase,
            $Command, '--memory-root', $FormalMemoryRoot, '--query', $Query,
            '--topic-limit', $TopicLimit, '--limit', $Limit, '--config', $ConfigPath
        )
        if ($ProjectDirectory) {
            $FormalRecallArguments += @('--project-directory', $ProjectDirectory)
        }
        if ($ProjectKey) {
            $FormalRecallArguments += @('--project-key', $ProjectKey)
        }
        if ($Domain) {
            $FormalRecallArguments += @('--domain', $Domain)
        }
        foreach ($Item in $Condition) {
            $FormalRecallArguments += @('--condition', $Item)
        }
        if ($ModelRouter) {
            $FormalRecallArguments += '--model-router'
        }
        & $PythonExecutable @FormalRecallArguments
    }
    elseif ($Command -eq 'session-memory') {
        & $PythonExecutable -m memory_lab --source-db $SourceDatabase --lab-db $LabDatabase `
            $Command --config $ConfigPath --memory-root $FormalMemoryRoot `
            --report-directory $ReportDirectory --settle-ms $SettleMilliseconds `
            --batch-size $BatchSize
    }
    elseif ($Command -in @('cycle', 'watch')) {
        $CycleArguments = @(
            '-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase,
            $Command, '--config', $ConfigPath, '--memory-root', $FormalMemoryRoot,
            '--report-directory', $ReportDirectory, '--settle-ms', $SettleMilliseconds
        )
        if ($Command -eq 'watch') {
            $CycleArguments += @('--interval-seconds', $IntervalSeconds)
            if ($MaxCycles -gt 0) {
                $CycleArguments += @('--max-cycles', $MaxCycles)
            }
        }
        if ($Refresh) {
            $CycleArguments += '--refresh'
        }
        & $PythonExecutable @CycleArguments
    }
    elseif ($Command -eq 'topics') {
        $TopicArguments = @(
            '-m', 'memory_lab', '--source-db', $SourceDatabase, '--lab-db', $LabDatabase,
            $Command
        )
        if ($PlanPath) {
            $TopicArguments += @('--apply-plan', $PlanPath)
        }
        & $PythonExecutable @TopicArguments
    }
    elseif ($Command -eq 'rollback-memory') {
        if (-not $MemoryId -or -not $TargetRevisionId -or -not $Reason) {
            throw 'rollback-memory requires -MemoryId, -TargetRevisionId, and -Reason'
        }
        & $PythonExecutable -m memory_lab --source-db $SourceDatabase --lab-db $LabDatabase `
            $Command --memory-id $MemoryId --target-revision-id $TargetRevisionId `
            --reason $Reason --requested-by 'user'
    }
    elseif ($Command -eq 'soak-start') {
        if (-not $Name) {
            throw 'soak-start requires -Name'
        }
        & $PythonExecutable -m memory_lab --source-db $SourceDatabase --lab-db $LabDatabase `
            $Command --name $Name
    }
    elseif ($Command -eq 'soak-status') {
        & $PythonExecutable -m memory_lab --source-db $SourceDatabase --lab-db $LabDatabase $Command
    }
    else {
        & $PythonExecutable -m memory_lab --source-db $SourceDatabase --lab-db $LabDatabase $Command
    }
    exit $LASTEXITCODE
}
finally {
    $env:PYTHONPATH = $PreviousPythonPath
}
