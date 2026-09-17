# Build the in-process engine (decision 0001, stage 4b) with a toolchain that
# actually works on this machine.
#
# The problem it solves: `C:\msys64\ucrt64\bin` is on the Windows PATH, so a
# plain `cargo build --features local-llama` picks up MSYS2's cmake, which then
# hands MSVC the MinGW headers in `C:\msys64\ucrt64\include`. The compile dies
# deep inside llama.cpp's vendored cpp-httplib with
#
#     corecrt.h(170): error C2061: syntax error: identifier '__UINTPTR_TYPE__'
#     winnt.h(144):   error C1189: No supported target architecture.
#
# which reads like a llama.cpp bug and is not one - MSVC is simply reading the
# wrong standard library. Nothing on the Rust side can fix that, because it is
# decided before rustc is ever invoked.
#
# So this script does three things, all scoped to its own process: drop MSYS2
# and MinGW from PATH, import the MSVC environment from vcvars64, and put the
# CMake that ships inside Build Tools ahead of anything else.
#
# Usage, from the repo root:
#   powershell -ExecutionPolicy Bypass -File src-core\build-local-llama.ps1
#   powershell -ExecutionPolicy Bypass -File src-core\build-local-llama.ps1 -Cuda -Release
#   powershell -ExecutionPolicy Bypass -File src-core\build-local-llama.ps1 -Command test
#   powershell -ExecutionPolicy Bypass -File src-core\build-local-llama.ps1 -Cuda -Release -Command run -Example bench-local-llama

param(
    # CUDA needs the toolkit as well as Build Tools. Off by default so the
    # script still works on a machine that only has the C++ workload.
    [switch]$Cuda,
    [switch]$Release,
    # check (fast, no codegen), clippy, build, test, or run an example.
    [ValidateSet('check', 'clippy', 'build', 'test', 'run')]
    [string]$Command = 'check',
    # Which example `-Command run` runs. The engine has no binary of its own,
    # so an example is the only way to drive it from this script.
    [string]$Example
)

if ($Command -eq 'run' -and -not $Example) {
    throw "-Command run needs -Example <name>, e.g. -Example bench-local-llama."
}

$ErrorActionPreference = 'Stop'

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found. Install the Visual Studio Build Tools C++ workload."
}
$vs = & $vswhere -latest -products * -property installationPath
if (-not $vs) { throw "No Visual Studio installation found." }

$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
$vsCMake = Join-Path $vs 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'
foreach ($p in @($vcvars, (Join-Path $vsCMake 'cmake.exe'))) {
    if (-not (Test-Path $p)) { throw "Missing $p - is the C++ CMake component installed?" }
}

# MSYS2 and MinGW out of the way BEFORE vcvars runs, so nothing it sets can be
# shadowed afterwards.
$env:PATH = (($env:PATH -split ';') | Where-Object {
    $_ -and ($_ -notmatch '(?i)msys64|mingw')
}) -join ';'

# vcvars64 is a batch file, so the only way to get its variables is to run it
# and read the environment it leaves behind.
& "$env:ComSpec" /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        Set-Item -Path "env:$($Matches[1])" -Value $Matches[2]
    }
}

# Again after vcvars: it prepends its own entries but keeps whatever was there.
$env:PATH = (($env:PATH -split ';') | Where-Object {
    $_ -and ($_ -notmatch '(?i)msys64|mingw')
}) -join ';'
$env:PATH = "$vsCMake;$env:PATH"

$cmake = (Get-Command cmake).Source
if ($cmake -match '(?i)msys64|mingw') {
    throw "cmake still resolves to $cmake - the PATH scrub did not take."
}

$feature = if ($Cuda) { 'local-llama-cuda' } else { 'local-llama' }
$args = @($Command, '-p', 'vessel-core', '--features', $feature)
if ($Example) { $args += @('--example', $Example) }
if ($Release) { $args += '--release' }

Write-Host "cmake:    $cmake"
Write-Host "cl:       $((Get-Command cl).Source)"
Write-Host "features: $feature"
Write-Host "cargo $($args -join ' ')"
Write-Host ''

& cargo @args
exit $LASTEXITCODE
