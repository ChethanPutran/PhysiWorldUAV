# =============================================================================
# Isaac Sim 4.1.0 + noVNC + OmniDrones
# =============================================================================
# This image provides:
#   - Isaac Sim 4.1.0 (headless, WebRTC capable)
#   - noVNC stack (Xvfb, x11vnc, websockify, openbox) for browser-based GUI
#   - OmniDrones (drone RL platform) with Isaac Lab v1.0.0
#   - Python 3.10 environment with all dependencies
#
# Build:
#   docker build -t isaac-sim-omnidrones:4.1.0 .
#
# Run:
#   docker run --name isaac-sim --gpus all --network=host -it \
#     -e "ACCEPT_EULA=Y" \
#     -v /data2/chethan/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
#     ... (other volume mounts) \
#     isaac-sim-omnidrones:4.1.0
# =============================================================================

FROM nvcr.io/nvidia/isaac-sim:4.1.0

# -----------------------------------------------------------------------------
# Base environment
# -----------------------------------------------------------------------------
# Disable interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive 

# Allow running Isaac Sim as root user
ENV OMNI_KIT_ALLOW_ROOT=1 

# Override GL version to 4.6 for compatibility with Isaac Sim
ENV MESA_GL_VERSION_OVERRIDE=4.6 

# Set the display number for Xvfb
ENV DISPLAY=:99 

# Set the path to Isaac Sim installation
ENV ISAACSIM_PATH=/isaac-sim 

# Set the timezone to Asia/Kolkata
ENV TZ=Asia/Kolkata 

# Use copy mode for uv to avoid symlink issues in Docker
ENV UV_LINK_MODE=copy 

# Ensure Python output is sent straight to terminal (no buffering)
ENV PYTHONUNBUFFERED=1 

# -----------------------------------------------------------------------------
# Install system dependencies: noVNC stack, GL libraries, build tools, git
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb \
        x11vnc \
        novnc \
        websockify \
        openbox \
        tint2 \
        libgl1-mesa-glx \
        libgl1-mesa-dri \
        mesa-utils \
        x11-utils \
        net-tools \
        procps \
        git \
        build-essential \
        cmake \
        wget \
        curl \
        ca-certificates \
        nano vim wmctrl xdotool \
    && rm -rf /var/lib/apt/lists/*



# -----------------------------------------------------------------------------
# Install uv (Python package manager)
# -----------------------------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx


# -----------------------------------------------------------------------------
# Install pip into Isaac Sim's bundled Python
# (some setup.py scripts invoke `python -m pip` on this interpreter)
# -----------------------------------------------------------------------------
RUN /isaac-sim/kit/python/bin/python3 -m ensurepip --upgrade

# -----------------------------------------------------------------------------
# Create Python 3.10 environment for OmniDrones
# -----------------------------------------------------------------------------
RUN uv venv --python 3.10 --seed /opt/omnidrones_env

# Make this environment the default for the container
ENV VIRTUAL_ENV=/opt/omnidrones_env
ENV PATH="/opt/omnidrones_env/bin:${PATH}"

# -----------------------------------------------------------------------------
# Install Python build dependencies required by Isaac Lab / OmniDrones
# -----------------------------------------------------------------------------
RUN uv pip install usd-core==23.11 lxml==4.9.4 tqdm xxhash

# -----------------------------------------------------------------------------
# Clone Isaac Lab and checkout the version compatible with Isaac Sim 4.1.0
# (v1.0.0 is the last release that supports Python 3.10 / Isaac Sim 4.1.0)
# -----------------------------------------------------------------------------
WORKDIR /workspace
RUN git clone https://github.com/isaac-sim/IsaacLab.git && \
    cd IsaacLab && \
    git checkout v1.0.0

# Install Isaac Lab using its native installer
RUN cd /workspace/IsaacLab && ./isaaclab.sh --install

# -----------------------------------------------------------------------------
# Clone OmniDrones and install it in editable mode
# -----------------------------------------------------------------------------
RUN git clone https://github.com/btx0424/OmniDrones.git && \
    cd OmniDrones && \
    uv pip install -e .

# -----------------------------------------------------------------------------
# Create activation script that sets PYTHONPATH for Isaac Sim / Isaac Lab
# (This replicates what OmniDrones' conda_setup scripts do, but for uv/venv)
# -----------------------------------------------------------------------------
RUN cat > /opt/omnidrones_env/bin/activate_omnidrones << 'EOF'
#!/bin/bash
export ISAACSIM_PATH="/isaac-sim"
export PYTHONPATH="$ISAACSIM_PATH:$PYTHONPATH"
export PYTHONPATH="/workspace/IsaacLab/source:$PYTHONPATH"
export PYTHONPATH="/workspace/OmniDrones:$PYTHONPATH"
EOF
RUN chmod +x /opt/omnidrones_env/bin/activate_omnidrones

# -----------------------------------------------------------------------------
# Copy the noVNC entrypoint script
# -----------------------------------------------------------------------------
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# -----------------------------------------------------------------------------
# Default command: start Xvfb, noVNC, and Isaac Sim
# -----------------------------------------------------------------------------
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]