param(
    [Parameter(Position = 0, Mandatory = $true)]
    [ValidateSet('baseline', 'ingest', 'prepare', 'coverage', 'admit', 'turns', 'episodes', 'turn-phases', 'evidence-pools', 'pg-ingest-turn', 'pg-show-turn', 'pg-quality-report', 'pg-writer-context', 'pg-get-turn-evidence', 'pg-list-memory-topics', 'pg-expand-pointer', 'pg-read-memory-detail', 'pg-read-topic-detail', 'pg-search-related-memories', 'pg-export-markdown', 'pg-writer-dry-run', 'pg-commit-plan', 'pg-writer-commit', 'pg-validate-memory-changes', 'pg-commit-memory-changes', 'pg-memory-agent-run', 'pg-memory-react-run', 'pg-memory-agent-batch', 'llm-smoke-test', 'review', 'consolidate', 'shadow', 'report', 'catalog', 'recall', 'markdown-preview', 'markdown-recall', 'auto-memory', 'formal-publish', 'change-report', 'formal-recall', 'cycle', 'session-memory', 'watch', 'engineering-candidates', 'topics', 'rollback-memory', 'soak-start', 'soak-status')]
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

    [string]$RawPointerId,

    [ValidateRange(1, 20000)]
    [int]$MaxChars = 4000,

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

    [ValidateRange(1, 20)]
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

    [string]$TargetRevisionId,

    [string]$Reason,

    [string]$Name,

    [string]$PlanPath,

    [switch]$ModelRouter,

    [ValidateSet('all', 'core', 'candidate', 'pointer', 'router')]
    [string]$EvidenceLayer = 'all',

    [ValidateSet('core', 'candidate', 'pointer')]
    [string[]]$Layer,

    [switch]$IncludeInactive,

    [switch]$Refresh,

    [switch]$DryRun
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

if (-not $PythonExecutable -or -not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw 'Python 3.11+ was not found. Set MEMORY_LAB_PYTHON to an absolute python.exe path.'
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
