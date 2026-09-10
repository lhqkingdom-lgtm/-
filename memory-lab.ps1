param(
    [Parameter(Position = 0, Mandatory = $true)]
    [ValidateSet('baseline', 'ingest', 'prepare', 'coverage', 'admit', 'turns', 'episodes', 'review', 'consolidate', 'shadow', 'report', 'catalog', 'recall', 'markdown-preview', 'markdown-recall', 'auto-memory', 'formal-publish', 'change-report', 'formal-recall', 'cycle', 'session-memory', 'watch', 'engineering-candidates', 'topics', 'rollback-memory', 'soak-start', 'soak-status')]
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

    [string]$GoldPath = "$PSScriptRoot\gold\bootstrap_v1.json",

    [string]$ReportDirectory = "$PSScriptRoot\reports",

    [string]$MemoryDirectory = "$PSScriptRoot\preview-memory",

    [string]$FormalMemoryRoot = "$PSScriptRoot\memory",

    [string]$ShadowRunId,

    [string]$RunId,

    [string]$Query,

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

    [string]$TargetRevisionId,

    [string]$Reason,

    [string]$Name,

    [string]$PlanPath,

    [switch]$ModelRouter,

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
