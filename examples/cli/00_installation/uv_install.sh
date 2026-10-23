#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Install uv if you haven't already
(
    installer=$(mktemp) || exit 1
    trap 'rm -f "$installer"' EXIT
    curl -LsSf https://github.com/astral-sh/uv/releases/download/0.9.18/uv-installer.sh -o "$installer" || exit 1
    if command -v sha256sum >/dev/null 2>&1; then
        echo "e27424708d1ac59cfc94e3f540a111f2edbb37bc8164febce8ee7fa5d5505c71  $installer" | sha256sum -c - || exit 1
    else
        echo "e27424708d1ac59cfc94e3f540a111f2edbb37bc8164febce8ee7fa5d5505c71  $installer" | shasum -a 256 -c - || exit 1
    fi
    sh "$installer"
) || exit 1

# Quick install (uses default PyTorch from PyPI)
uv pip install anomalib

# To ensure a specific hardware backend, use one of the following extras:

# CPU support (works on all platforms)
uv pip install "anomalib[cpu]"

# CUDA support (Linux/Windows with NVIDIA GPU)
uv pip install "anomalib[cu126]"  # CUDA 12.6
uv pip install "anomalib[cu130]"  # CUDA 13.0

# ROCm support (Linux with AMD GPU)
uv pip install "anomalib[rocm]"

# Intel XPU support (Linux/Windows with Intel GPU)
uv pip install "anomalib[xpu]"

# You can combine extras. For example, to install with CUDA 13.0 and OpenVINO support:
uv pip install "anomalib[openvino,cu130]"

# For a full installation with all optional dependencies on CPU:
uv pip install "anomalib[full,cpu]"

# To install from source for development:
git clone https://github.com/open-edge-platform/anomalib.git
cd anomalib

# Create a virtual environment and sync with the lockfile for a specific backend:
uv venv
uv sync --extra cpu

# Or for CUDA 13.0
uv sync --extra cu130

# For a full development environment on CPU
uv sync --extra dev --extra cpu
