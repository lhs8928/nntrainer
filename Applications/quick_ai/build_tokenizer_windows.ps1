# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2026 The NNTrainer Authors
#
# @file build_tokenizer_windows.ps1
# @brief Build the CausalLM tokenizers_c library for Windows.

param (
    [string]$BuildDir = "build",
    [string]$RustTarget = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$CrateDir = Join-Path $ScriptDir "tokenizers_c_win"
$CargoBin = Join-Path $env:USERPROFILE ".cargo\bin"

if (-Not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    $CargoExe = Join-Path $CargoBin "cargo.exe"
    if (Test-Path $CargoExe) {
        $env:PATH = "$CargoBin;$env:PATH"
    } else {
        Write-Error "cargo is required. Install Rust from https://rustup.rs/ and retry."
    }
}

if ([System.IO.Path]::IsPathRooted($BuildDir)) {
    $BuildRoot = $BuildDir
} else {
    $BuildRoot = Join-Path $RepoRoot $BuildDir
}

$TargetDir = Join-Path $BuildRoot "tokenizers_c_win\target"
$CargoArgs = @(
    "build",
    "--manifest-path", (Join-Path $CrateDir "Cargo.toml"),
    "--target-dir", $TargetDir,
    "--release",
    "--locked"
)

if (-Not [string]::IsNullOrWhiteSpace($RustTarget)) {
    if (Get-Command rustup -ErrorAction SilentlyContinue) {
        & rustup target add $RustTarget
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }

    $CargoArgs += @("--target", $RustTarget)
}

New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null

Write-Output "Building tokenizers_c for Windows"
Write-Output "  crate:  $CrateDir"
Write-Output "  target: $TargetDir"

$PreviousRustFlags = $env:RUSTFLAGS
$PreviousCFlags = $env:CFLAGS
$PreviousCxxFlags = $env:CXXFLAGS
$DynamicCrtFlag = "-C target-feature=-crt-static"

try {
    if ([string]::IsNullOrWhiteSpace($PreviousRustFlags)) {
        $env:RUSTFLAGS = $DynamicCrtFlag
    } else {
        $env:RUSTFLAGS = "$PreviousRustFlags $DynamicCrtFlag"
    }

    if ([string]::IsNullOrWhiteSpace($PreviousCFlags)) {
        $env:CFLAGS = "/MD"
    } else {
        $env:CFLAGS = "$PreviousCFlags /MD"
    }

    if ([string]::IsNullOrWhiteSpace($PreviousCxxFlags)) {
        $env:CXXFLAGS = "/MD"
    } else {
        $env:CXXFLAGS = "$PreviousCxxFlags /MD"
    }

    & cargo @CargoArgs
} finally {
    $env:RUSTFLAGS = $PreviousRustFlags
    $env:CFLAGS = $PreviousCFlags
    $env:CXXFLAGS = $PreviousCxxFlags
}

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ([string]::IsNullOrWhiteSpace($RustTarget)) {
    $LibraryPath = Join-Path $TargetDir "release\tokenizers_c.lib"
} else {
    $LibraryPath = Join-Path $TargetDir "$RustTarget\release\tokenizers_c.lib"
}

if (-Not (Test-Path $LibraryPath)) {
    Write-Error "tokenizers_c.lib was not produced at $LibraryPath"
}

Write-Output "Built $LibraryPath"
