$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $scriptDir '.venv\Scripts\python.exe'
$pythonCmd = $null
$requiredImports = 'import numpy,pandas,scipy,matplotlib,openpyxl,torch'

function Test-PythonEnv([string]$pythonExe, [string]$imports) {
    if (-not $pythonExe) { return $false }
    & $pythonExe -c $imports *> $null
    return ($LASTEXITCODE -eq 0)
}

if ($env:CONDA_PREFIX) {
    $condaPython = Join-Path $env:CONDA_PREFIX 'python.exe'
    if (Test-PythonEnv $condaPython $requiredImports) {
        $pythonCmd = $condaPython
    }
}

if (-not $pythonCmd -and (Test-Path $venvPython)) {
    if (Test-PythonEnv $venvPython $requiredImports) {
        $pythonCmd = $venvPython
    }
}

if (-not $pythonCmd) {
    if (Test-PythonEnv 'python' $requiredImports) {
        $pythonCmd = 'python'
    }
}

if (-not $pythonCmd) {
    Write-Host '未找到可直接运行的Python环境（缺少项目依赖）。'
    Write-Host '建议在已存在的conda环境中补齐最小依赖后重试：'
    Write-Host 'conda activate ai_env'
    Write-Host 'pip install torch xlrd'
    exit 1
}

Write-Host "使用解释器: $pythonCmd"

$env:DATASET_A_DIR = Join-Path $scriptDir 'GCMS_单个样本数据'
$env:OUTPUT_DIR = Join-Path $scriptDir 'outputs\cnn_baseline'

& $pythonCmd (Join-Path $scriptDir 'CNN.py')
