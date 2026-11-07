# HoreKa2 environment for this project (source it). Everything lives in the shared workspace, nothing in $HOME
# (home quota): uv binary, uv cache, uv-managed Python builds and other caches.
export P=/path/to/workspace/transferable_excited_MLIP
export UV_CACHE_DIR=$P/.uv/cache UV_PYTHON_INSTALL_DIR=$P/.uv/python UV_TOOL_DIR=$P/.uv/tools
export XDG_CACHE_HOME=$P/.cache MPLCONFIGDIR=$P/.cache/matplotlib PIP_CACHE_DIR=$P/.cache/pip
export PATH=$P/.uv/bin:$PATH
export MODELSPATH=$P/models   # MLatom model weights (OMNI-P2x) in the workspace, not in ~/.mlatom
