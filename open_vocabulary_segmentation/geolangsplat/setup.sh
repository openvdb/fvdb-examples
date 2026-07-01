#!/usr/bin/env bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# One-shot setup for GeoLangSplat (e.g. when baking a DLI lab image).
#
# Run ONCE at image-build time, from inside the env that already has fvdb + torch +
# fvdb-reality-capture (the same env the reconstruction / mesh-extraction lab uses).
# It downloads the SAM3 weights, installs GeoLangSplat, and runs the preflight so the
# image ships ready -- the lab notebook then only needs to `import` and run `gls doctor`.
#
# Configure via env vars (only the URL is site-specific):
#   GEOLANGSPLAT_SAM_URL   download URL for the SAM3 checkpoint (required to auto-download)
#   GEOLANGSPLAT_SAM_CKPT  where to place / find the checkpoint
#                          (default: ./sam3_ckpt/sam3.1_multiplex.pt)
#   GEOLANGSPLAT_SAM_REPO  optional git URL for the SAM3 package (pip-installed if `sam3`
#                          is not already importable)
#
# Usage:
#   GEOLANGSPLAT_SAM_URL=https://.../sam3.1_multiplex.pt ./setup.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT="${GEOLANGSPLAT_SAM_CKPT:-${HERE}/sam3_ckpt/sam3.1_multiplex.pt}"

# --- 1. SAM3 weights -------------------------------------------------------
# Large + gated, so they must live on disk before the lab. Download once if a URL
# is given and the file isn't already there.
if [[ -f "${CKPT}" ]]; then
  echo "[setup] SAM3 checkpoint already present: ${CKPT}"
elif [[ -n "${GEOLANGSPLAT_SAM_URL:-}" ]]; then
  echo "[setup] downloading SAM3 checkpoint -> ${CKPT}"
  mkdir -p "$(dirname "${CKPT}")"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -o "${CKPT}" "${GEOLANGSPLAT_SAM_URL}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${CKPT}" "${GEOLANGSPLAT_SAM_URL}"
  else
    echo "[setup] FATAL: neither curl nor wget available to download the checkpoint" >&2
    exit 1
  fi
else
  echo "[setup] no checkpoint at ${CKPT} and GEOLANGSPLAT_SAM_URL is unset --"
  echo "[setup] set GEOLANGSPLAT_SAM_URL to auto-download, or copy the weights into place."
fi
export GEOLANGSPLAT_SAM_CKPT="${CKPT}"

# --- 2. SAM3 package -------------------------------------------------------
# Not on PyPI. Install from source if a repo URL is given and it isn't importable yet.
if python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('sam3') else 1)"; then
  echo "[setup] sam3 package already importable"
elif [[ -n "${GEOLANGSPLAT_SAM_REPO:-}" ]]; then
  echo "[setup] installing SAM3 package from ${GEOLANGSPLAT_SAM_REPO}"
  pip install "git+${GEOLANGSPLAT_SAM_REPO}"
else
  echo "[setup] sam3 not importable and GEOLANGSPLAT_SAM_REPO unset -- install SAM3 from source into the image."
fi

# --- 3. GeoLangSplat (light) ----------------------------------------------
echo "[setup] installing GeoLangSplat (editable) from ${HERE}"
pip install -e "${HERE}"

# --- 4. Preflight ----------------------------------------------------------
echo "[setup] preflight:"
gls doctor || echo "[setup] doctor reported issues above -- fix them in the image before shipping the lab."

echo "[setup] done. Persist GEOLANGSPLAT_SAM_CKPT=${CKPT} in the image env."
echo "[setup] In the notebook, the first cell only needs: import geolangsplat + gls doctor (no installs)."
