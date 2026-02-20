#!/bin/bash
# RAG Pipeline Skill Installer for Claude Code
# Usage: curl -fsSL https://raw.githubusercontent.com/etlab8320/Ragskill/v1.1.0/install.sh | bash
set -euo pipefail

VERSION="1.1.0"
REPO="etlab8320/Ragskill"
BASE_URL="https://raw.githubusercontent.com/${REPO}/v${VERSION}"
SKILL_DIR="$HOME/.claude/skills/rag-pipeline"
CHECKSUM_URL="${BASE_URL}/checksums.sha256"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# --uninstall
if [[ "${1:-}" == "--uninstall" ]]; then
    if [[ -d "$SKILL_DIR" ]]; then
        rm -rf "$SKILL_DIR"
        info "RAG Pipeline Skill uninstalled from $SKILL_DIR"
    else
        warn "Not installed at $SKILL_DIR"
    fi
    exit 0
fi

# Check: Claude Code CLI
if ! command -v claude &>/dev/null; then
    error "Claude Code CLI not found."
    error "Install from https://docs.anthropic.com/en/docs/claude-code"
    exit 1
fi

# Check: existing installation
if [[ -f "$SKILL_DIR/SKILL.md" ]]; then
    warn "Existing installation detected at $SKILL_DIR"
    read -rp "Overwrite? [y/N] " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        info "Installation cancelled."
        exit 0
    fi
fi

info "Installing RAG Pipeline Skill v${VERSION}..."

mkdir -p "$SKILL_DIR"

# Download SKILL.md
curl -fsSL "${BASE_URL}/skill/SKILL.md" -o "$SKILL_DIR/SKILL.md"

# SHA256 verification
EXPECTED_SHA=$(curl -fsSL "$CHECKSUM_URL" 2>/dev/null | grep "SKILL.md" | awk '{print $1}') || true
if [[ -n "$EXPECTED_SHA" ]]; then
    ACTUAL_SHA=$(sha256sum "$SKILL_DIR/SKILL.md" | awk '{print $1}')
    if [[ "$EXPECTED_SHA" != "$ACTUAL_SHA" ]]; then
        error "Checksum verification failed!"
        error "Expected: $EXPECTED_SHA"
        error "Actual:   $ACTUAL_SHA"
        rm -f "$SKILL_DIR/SKILL.md"
        exit 1
    fi
    info "Checksum verified."
else
    warn "Checksum file not found at v${VERSION}. Skipping verification."
fi

echo ""
info "RAG Pipeline Skill v${VERSION} installed successfully!"
echo "  Location: $SKILL_DIR/SKILL.md"
echo ""
echo "Usage: Open Claude Code and type /rag-pipeline"
echo ""
echo "Prerequisites:"
echo "  - VOYAGE_API_KEY (https://dash.voyageai.com/ - free 50M tokens/month)"
echo "  - LLM provider: Gemini (default) / Claude CLI / OpenAI / Claude API"
echo ""
echo "Uninstall: curl -fsSL ${BASE_URL}/install.sh | bash -s -- --uninstall"
