#!/bin/bash
set -e

# ── Environment ────────────────────────────────────────────────
export DISPLAY=${DISPLAY:-:99}
export SCREEN_RES=${SCREEN_RES:-1920x1080x24}
export VNC_PORT=${VNC_PORT:-5900}
export NOVNC_PORT=${NOVNC_PORT:-6080}
export OMNI_KIT_ALLOW_ROOT=1
export MESA_GL_VERSION_OVERRIDE=4.6
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia

echo "=============================================="
echo " Starting Isaac Sim + OmniDrones container"
echo "   DISPLAY    = $DISPLAY"
echo "   SCREEN_RES = $SCREEN_RES"
echo "   VNC_PORT   = $VNC_PORT"
echo "   NOVNC_PORT = $NOVNC_PORT"
echo "=============================================="

# ── 1. Xvfb ────────────────────────────────────────────────────
if ! pgrep -x Xvfb > /dev/null; then
    echo "[1/6] Starting Xvfb on $DISPLAY ..."
    Xvfb "$DISPLAY" -screen 0 "$SCREEN_RES" -ac +extension GLX +render -noreset &
    sleep 2
    xdpyinfo -display "$DISPLAY" > /dev/null 2>&1 || { echo "Xvfb failed"; exit 1; }
else
    echo "[1/6] Xvfb already running"
fi

# ── 2. Openbox (window manager) ────────────────────────────────
if ! pgrep -x openbox > /dev/null; then
    echo "[2/6] Starting Openbox ..."
    openbox --sm-disable &
    sleep 1
else
    echo "[2/6] Openbox already running"
fi

# ── 3. tint2 (taskbar so minimized windows can be restored) ────
if ! pgrep -x tint2 > /dev/null; then
    echo "[3/6] Starting tint2 taskbar ..."
    tint2 > /dev/null 2>&1 &
fi

# ── 4. x11vnc ──────────────────────────────────────────────────
if ! pgrep -x x11vnc > /dev/null; then
    echo "[4/6] Starting x11vnc on port $VNC_PORT ..."
    x11vnc -display "$DISPLAY" -forever -nopw -shared -noxdamage \
           -rfbport "$VNC_PORT" > /var/log/x11vnc.log 2>&1 &
    sleep 1
else
    echo "[4/6] x11vnc already running"
fi

# ── 5. websockify (noVNC web bridge) ───────────────────────────
if ! pgrep -f websockify > /dev/null; then
    echo "[5/6] Starting noVNC on port $NOVNC_PORT ..."
    websockify --web /usr/share/novnc/ "$NOVNC_PORT" "localhost:$VNC_PORT" \
               > /var/log/websockify.log 2>&1 &
    sleep 1
else
    echo "[5/6] websockify already running"
fi

# ── 6. Activate the WORKING venv (on the bind mount) ───────────
# NOTE: the image's ENV still has VIRTUAL_ENV=/opt/omnidrones_env.
# That venv is from the old Dockerfile build and is not the one you
# actually use. Override it here so the shell gets the right Python.
unset VIRTUAL_ENV
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/opt/omnidrones_env/bin' | paste -sd:)

if [ -d /workspace/PhysiWorldUAV/omnidrones_env ]; then
    echo "[6/6] Activating /workspace/PhysiWorldUAV/omnidrones_env ..."
    # shellcheck disable=SC1091
    source /workspace/PhysiWorldUAV/omnidrones_env/bin/activate
else
    echo "[6/6] WARNING: /workspace/PhysiWorldUAV/omnidrones_env not found"
    echo "      Make sure the /workspace/PhysiWorldUAV bind mount is present."
fi

echo ""
echo "=============================================="
echo " Container ready.  No Isaac Sim running yet."
echo ""
echo " From your laptop:"
echo "   ssh -L $NOVNC_PORT:localhost:$NOVNC_PORT <user>@<server_ip>"
echo "   Open http://localhost:$NOVNC_PORT/vnc.html"
echo ""
echo " Inside the container:"
echo "   cd /workspace/PhysiWorldUAV/scripts"
echo "   python manual_flight.py    # spawns ONE Isaac Sim instance"
echo "   python collect_data.py     # spawns ONE Isaac Sim instance"
echo "=============================================="
echo ""

# ── Drop to shell or run the provided command ──────────────────
if [ $# -gt 0 ]; then
    exec "$@"
else
    exec bash
fi