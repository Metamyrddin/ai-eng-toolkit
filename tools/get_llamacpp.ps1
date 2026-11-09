# This script downloads the official llama.cpp programs for Windows into the .bin folder.
# It uses the CPU zip from ggml-org/llama.cpp. The Linux tar file does not run on Windows.
# The health check looks for llama-server.exe inside that download.
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
# Older Windows PowerShell needs this line before it will open https addresses.
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
} catch {
    Write-Host "Could not set TLS 1.2. The download will still be tried."
}

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$bin = Join-Path $root ".bin"
if (-not (Test-Path $bin)) {
    New-Item -ItemType Directory -Path $bin | Out-Null
}

$headers = @{
    "User-Agent" = "ai-eng-toolkit"
    "Accept"     = "application/vnd.github+json"
}
$releases = Invoke-RestMethod -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10" -Headers $headers
$url = $null
foreach ($rel in @($releases)) {
    foreach ($asset in @($rel.assets)) {
        # The x64 CPU zip only. Do not take the ARM zip or a CUDA zip.
        if ($asset.name -like "*bin-win-cpu-x64.zip") {
            $url = $asset.browser_download_url
            break
        }
    }
    if ($url) { break }
}
if (-not $url) {
    throw "Could not find a Windows CPU llama.cpp zip in the recent GitHub releases."
}

Write-Host "downloading $url"
Set-Content -Path (Join-Path $bin "VERSION_URL.txt") -Value $url -Encoding ascii
$zip = Join-Path $bin "llama-win.zip"
Invoke-WebRequest -Uri $url -OutFile $zip -Headers @{ "User-Agent" = "ai-eng-toolkit" }
$zipName = [System.IO.Path]::GetFileNameWithoutExtension(([Uri]$url).AbsolutePath.Split("/")[-1])
$dest = Join-Path $bin $zipName
if (Test-Path $dest) {
    Remove-Item -Recurse -Force $dest
}
Expand-Archive -Path $zip -DestinationPath $dest -Force
Remove-Item -Force $zip
Get-ChildItem -Recurse -File $dest | Select-Object -ExpandProperty FullName
