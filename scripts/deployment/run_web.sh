#!/bin/bash
# scripts/deployment/run_web.sh

echo "Starting React Web interface (Vite)..."

# Check Node.js
if ! command -v npm &> /dev/null; then
    echo "Node.js / npm is not installed!"
    exit 1
fi

# Start Web
cd "$(dirname "$0")/../../behavior-detection-web" || exit
npm run dev
