#!/bin/bash
set -e

PACKAGE_DIR="package"
ZIP_FILE="lambda.zip"

echo "==> Cleaning previous build..."
rm -rf "$PACKAGE_DIR" "$ZIP_FILE"
mkdir -p "$PACKAGE_DIR"

echo "==> Installing dependencies (Linux x86_64 / Python 3.12 wheels)..."
pip3 install \
  --platform manylinux2014_x86_64 \
  --python-version 3.12 \
  --only-binary=:all: \
  --implementation cp \
  --target "$PACKAGE_DIR" \
  -r requirements.txt

echo "==> Stripping unused large packages..."
# hf_xet: XET storage protocol for model downloads — not needed for Inference API calls
rm -rf "$PACKAGE_DIR/hf_xet" "$PACKAGE_DIR"/hf_xet-*.dist-info

echo "==> Copying source files..."
cp server.py rag.py embeddings.py pinecone_client.py \
   settings.py utils.py llm_rerank.py lambda_handler.py "$PACKAGE_DIR/"

echo "==> Zipping..."
rm -f "$ZIP_FILE"
cd "$PACKAGE_DIR"
zip -r "../$ZIP_FILE" . -x "*.pyc" -x "__pycache__/*"
cd ..

SIZE=$(du -sh "$ZIP_FILE" | cut -f1)
echo "==> Done: $ZIP_FILE ($SIZE)"
