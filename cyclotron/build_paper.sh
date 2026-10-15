#!/usr/bin/env bash
set -euo pipefail

# Directory setup
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LATEX_DIR="${PROJECT_DIR}/latex"
OUTPUT_DIR="${PROJECT_DIR}/output"
ZIP_NAME="ds_dualmode_paper_ieee.zip"

echo "=== [1/4] Preparing Output Directories ==="
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LATEX_DIR}/figures"

echo "=== [2/4] Compiling IEEE LaTeX Document ==="
cd "${LATEX_DIR}"

# Run pdflatex compilation sequence
pdflatex -interaction=nonstopmode -output-directory="${OUTPUT_DIR}" main.tex
if [ -f "references.bib" ]; then
    bibtex "${OUTPUT_DIR}/main" || true
    pdflatex -interaction=nonstopmode -output-directory="${OUTPUT_DIR}" main.tex
fi
pdflatex -interaction=nonstopmode -output-directory="${OUTPUT_DIR}" main.tex

echo "=== [3/4] Validating PDF Output ==="
if [ -f "${OUTPUT_DIR}/main.pdf" ]; then
    echo "[SUCCESS] PDF compiled successfully: ${OUTPUT_DIR}/main.pdf"
else
    echo "[ERROR] PDF compilation failed!"
    exit 1
fi

echo "=== [4/4] Creating Complete ZIP Archive ==="
cd "${PROJECT_DIR}"
zip -r "${OUTPUT_DIR}/${ZIP_NAME}" latex/ build_paper.sh -x "*.aux" "*.log" "*.out" "*.toc"

echo "=== Artifacts Generated Successfully ==="
echo "PDF Document: ${OUTPUT_DIR}/main.pdf"
echo "ZIP Archive:  ${OUTPUT_DIR}/${ZIP_NAME}"