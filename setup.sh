#!/usr/bin/env bash
# chewing-vision quick setup
# Usage: bash setup.sh
set -euo pipefail

CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'; RED='\033[0;31m'; GREEN='\033[0;32m'
info()    { echo -e "  ${CYAN}→${RESET}  $*"; }
success() { echo -e "  ${GREEN}✓${RESET}  $*"; }
error()   { echo -e "  ${RED}✗${RESET}  $*" >&2; exit 1; }

echo ""
echo -e "  ${BOLD}${CYAN}chewing-vision${RESET}  setup"
echo -e "  ${CYAN}──────────────────────────────${RESET}"
echo ""

# ── 1. Python version check ──────────────────────────────────────────────────
PYTHON=$(command -v python3 || true)
[[ -z "$PYTHON" ]] && error "python3 not found. Install Python 3.10+ first."

PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_OK=$("$PYTHON" -c "import sys; print('yes' if sys.version_info >= (3,10) else 'no')")
[[ "$PY_OK" != "yes" ]] && error "Python 3.10+ required (found $PY_VER)."
info "Python $PY_VER  ✓"

# ── 2. Virtual environment ───────────────────────────────────────────────────
if [[ ! -d ".venv" ]]; then
    info "Creating .venv …"
    "$PYTHON" -m venv .venv
fi
success ".venv ready"

PIP=".venv/bin/pip"
"$PIP" install --upgrade pip -q

# ── 3. Install package ───────────────────────────────────────────────────────
info "Installing chewing-vision + firebase …"
"$PIP" install -e ".[firebase]" -q
success "Package installed"

# ── 4. Global command symlink ────────────────────────────────────────────────
BIN=".venv/bin/chewing-vision"
LINK="/usr/local/bin/chewing-vision"

if [[ ! -f "$BIN" ]]; then
    error "Entry point not found at $BIN — install may have failed."
fi

if [[ -L "$LINK" || -f "$LINK" ]]; then
    info "Updating symlink $LINK"
    sudo ln -sf "$(pwd)/$BIN" "$LINK"
else
    info "Creating symlink $LINK (sudo password may be needed)"
    sudo ln -sf "$(pwd)/$BIN" "$LINK"
fi
success "Global command: chewing-vision"

# ── 5. .env for Firebase credentials ────────────────────────────────────────
if [[ ! -f ".env" ]]; then
    info "Creating .env template …"
    printf '# Firebase service account JSON path.\n# Get it: Firebase Console → Project Settings → Service Accounts → Generate new private key\nCHEWING_FIREBASE_CREDENTIALS=\n' > .env
    echo ""
    echo -e "  ${RED}!${RESET}  Fill in CHEWING_FIREBASE_CREDENTIALS in .env to enable firebase fetch."
else
    success ".env already exists — skipped"
fi

echo ""
echo -e "  ${GREEN}${BOLD}All done.${RESET}  Run:  ${CYAN}chewing-vision${RESET}"
echo ""
