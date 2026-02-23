#!/bin/bash
# ══════════════════════════════════════════════════════════════
# ARGUS — One-Command Demo Environment Setup
# Sets up everything needed for the 7-minute hackathon demo
# ══════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "════════════════════════════════════════════════════════"
echo "  🌊 ARGUS — Demo Environment Setup"
echo "════════════════════════════════════════════════════════"
echo ""

# ── Step 1: Python dependencies ──────────────────────────
echo "📦 Installing Python dependencies..."
pip install -q -r requirements.txt 2>/dev/null
echo "  ✅ Python dependencies installed"

# ── Step 2: Dashboard dependencies ───────────────────────
echo "📦 Installing Dashboard dependencies..."
cd dashboard
if [ ! -d "node_modules" ]; then
    npm install --silent 2>/dev/null
fi
cd "$PROJECT_ROOT"
echo "  ✅ Dashboard dependencies installed"

# ── Step 3: Generate synthetic Sentinel tiles ────────────
echo "🛰️  Generating synthetic Sentinel-2 tiles for ScarNet demo..."
python scripts/generate_synthetic_sentinel_tiles.py 2>/dev/null || echo "  ⚠️  Tile generation skipped (rasterio not installed — ScarNet will use numpy fallback)"

# ── Step 4: Create model directories ────────────────────
echo "📁 Ensuring model directories exist..."
mkdir -p models data/sentinel2 data/dags
echo "  ✅ Directories ready"

# ── Step 5: Verify .env ─────────────────────────────────
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "  ✅ .env created from .env.example"
    else
        echo "  ⚠️  No .env file found — services will use defaults"
    fi
else
    echo "  ✅ .env exists"
fi

# ── Step 6: Docker infrastructure ───────────────────────
echo "🐳 Starting infrastructure containers..."
if command -v docker &>/dev/null; then
    docker compose up -d 2>/dev/null && echo "  ✅ Kafka, TimescaleDB, Redis running" || echo "  ⚠️  Docker Compose failed — demo mode will work"
else
    echo "  ⚠️  Docker not available — services run in demo mode"
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ✅ Setup complete!"
echo ""
echo "  Next steps:"
echo "    ./scripts/start_all.sh          # Start all services"
echo "    ./scripts/run_demo_scenario.sh  # Run the 7-minute demo"
echo "════════════════════════════════════════════════════════"
