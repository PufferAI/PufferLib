# PufferLib Windows build wrapper over the CMake presets.
#
# Usage:
#   .\build.ps1 breakout            # CUDA _C extension (default)
#   .\build.ps1 breakout -Cpu       # CPU-only _C extension
#   .\build.ps1 breakout -Float     # float32 precision
#   .\build.ps1 breakout -DebugBuild # Debug build
#   .\build.ps1 breakout -Fast      # Standalone optimized executable
#   .\build.ps1 breakout -Local     # Standalone debug executable
#   .\build.ps1 breakout -FetchCudnn # Auto-download cuDNN (~1.7 GB, one-time)
#
# Requirements: CMake + Ninja, LLVM clang, Visual Studio Build Tools (cl.exe),
# CUDA toolkit (CUDA_PATH), and cuDNN for the CUDA extension (see README).

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$EnvName,
    [switch]$Cpu,
    [switch]$Float,
    [switch]$DebugBuild,
    [switch]$Fast,
    [switch]$Local,
    [switch]$FetchCudnn
)

$ErrorActionPreference = 'Stop'

# Make sure the MSVC environment (cl.exe, link.exe, Windows SDK) is loaded.
# For CUDA builds the environment must come from a VS2022 (v143) toolset:
# CUDA 12.x nvcc cannot parse VS2026 (v14.5x) headers. CPU/standalone builds
# work with any MSVC.
$needV143 = -not ($Cpu -or $Fast -or $Local)
$vcvars = $env:VCVARS
if (-not $vcvars) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $range = if ($needV143) { '[17.0,18.0)' } else { '[17.0,)' }
        $vsroot = & $vswhere -latest -products * -version $range -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($vsroot) { $vcvars = Join-Path @($vsroot)[0] 'VC\Auxiliary\Build\vcvars64.bat' }
    }
}
if ((-not $vcvars -or -not (Test-Path $vcvars)) -and (Get-Command cl -ErrorAction SilentlyContinue)) {
    # Already in a developer prompt; trust it (CUDA builds may still fail on VS2026)
    $vcvars = $null
} elseif (-not $vcvars -or -not (Test-Path $vcvars)) {
    if ($needV143) {
        throw "No VS2022 (v143) toolset found. CUDA 12.x requires it: install 'Visual Studio 2022 Build Tools' with the C++ workload, or set VCVARS to its vcvars64.bat. (Alternatively upgrade to CUDA >= 13.2 for VS2026 support.)"
    }
    throw "cl.exe not found and vcvars64.bat could not be located. Run from a 'x64 Native Tools' prompt or set VCVARS to your vcvars64.bat path."
}
if ($vcvars) {
    Write-Host "Loading MSVC environment from $vcvars"
    $envDump = cmd /c "`"$vcvars`" >nul 2>&1 && set"
    foreach ($line in $envDump) {
        if ($line -match '^([^=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
        }
    }
}

$preset = 'windows-cuda'
if ($Cpu)   { $preset = 'windows-cpu' }
if ($Fast)  { $preset = 'windows-fast' }
if ($Local) { $preset = 'windows-local' }
if ($DebugBuild -and $preset -eq 'windows-cuda') { $preset = 'windows-cuda-debug' }

$extra = @()
if ($Float) { $extra += '-DPRECISION_FLOAT=ON' }
if ($DebugBuild -and $preset -ne 'windows-cuda-debug') { $extra += '-DCMAKE_BUILD_TYPE=Debug' }
if ($env:NVCC_ARCH) { $extra += "-DCMAKE_CUDA_ARCHITECTURES=$($env:NVCC_ARCH)" }
if ($FetchCudnn) { $extra += '-DPUFFER_FETCH_CUDNN=ON' }

cmake --preset $preset -DENV="$EnvName" @extra
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }
cmake --build --preset $preset
if ($LASTEXITCODE -ne 0) { throw "Build failed" }
