#!/bin/bash
# RAG Pipeline Skill Installer for Claude Code
# Usage: curl -fsSL https://raw.githubusercontent.com/etlab8320/Ragskill/main/install.sh | bash

set -e

SKILL_DIR="$HOME/.claude/skills/rag-pipeline"

echo "Installing RAG Pipeline Skill..."

mkdir -p "$SKILL_DIR"

# Download SKILL.md
curl -fsSL "https://raw.githubusercontent.com/etlab8320/Ragskill/main/skill/SKILL.md" \
  -o "$SKILL_DIR/SKILL.md"

echo ""
echo "RAG Pipeline Skill installed successfully!"
echo "Location: $SKILL_DIR/SKILL.md"
echo ""
echo "Usage: Open Claude Code and type /rag-pipeline"
echo ""
echo "Prerequisites:"
echo "  - VOYAGE_API_KEY (https://dash.voyageai.com/ - free 50M tokens/month)"
echo "  - ANTHROPIC_API_KEY (https://console.anthropic.com/)"
