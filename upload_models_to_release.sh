#!/bin/bash
# Upload production models to GitHub Release so GitHub Actions can download them
# without your MacBook being open.
#
# Run this AFTER training completes and AFTER you've copied the best w2 models
# into models-live/:
#   cp -r models/PPO_w2_s42 models-live/
#   cp -r models/A2C_w2_s42 models-live/
#   cp -r models/SAC_w2_s42 models-live/
#   bash upload_models_to_release.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TAG="v1.0-models"
TARBALL="models-live.tar.gz"

echo "=== Sector Command — Upload Models to GitHub Release ==="
echo ""

# Verify models-live/ exists and has the three agents
for agent in A2C PPO SAC; do
    found=$(ls "$SCRIPT_DIR/models-live/" | grep "^${agent}_" | head -1)
    if [ -z "$found" ]; then
        echo "ERROR: No ${agent} model found in models-live/. Run training first."
        exit 1
    fi
    echo "  ✓ Found $agent: $found"
done
echo ""

# Package into a single tarball
echo "Packaging models-live/ → $TARBALL ..."
cd "$SCRIPT_DIR"
tar -czf "/tmp/$TARBALL" models-live/
SIZE=$(du -sh "/tmp/$TARBALL" | cut -f1)
echo "  Size: $SIZE"
echo ""

# Delete old release assets and re-upload
echo "Uploading to GitHub Release '$TAG'..."
# If release doesn't exist yet, create it
if ! gh release view "$TAG" --repo cameroncc333/rl-portfolio-optimizer >/dev/null 2>&1; then
    echo "  Creating new release '$TAG'..."
    gh release create "$TAG" \
        --repo cameroncc333/rl-portfolio-optimizer \
        --title "Production RL Models" \
        --notes "Production inference models for Sector Command Live. Updated automatically after retraining." \
        "/tmp/$TARBALL"
else
    echo "  Updating existing release '$TAG'..."
    # Delete old asset if it exists, then upload fresh
    gh release delete-asset "$TAG" "$TARBALL" \
        --repo cameroncc333/rl-portfolio-optimizer \
        --yes 2>/dev/null || true
    gh release upload "$TAG" "/tmp/$TARBALL" \
        --repo cameroncc333/rl-portfolio-optimizer \
        --clobber
fi

echo ""
echo "=== Done ==="
echo "  Models live at: https://github.com/cameroncc333/rl-portfolio-optimizer/releases/tag/$TAG"
echo "  GitHub Actions will download them automatically before each briefing."
echo ""
echo "  Next steps after retraining:"
echo "  1. Copy best w2 models:  cp -r models/PPO_w2_s42 models-live/"
echo "                           cp -r models/A2C_w2_s42 models-live/"
echo "                           cp -r models/SAC_w2_s42 models-live/"
echo "  2. Run this script again to update the release."
