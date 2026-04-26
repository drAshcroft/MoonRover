# Moon Rover Installation Guide

This guide covers setting up the Moon Rover simulation environment for development and execution.

## System Requirements

### Python
- **Python 3.10 or later** required
- Recommended: Python 3.11 or 3.12 for best Genesis engine performance

### GPU and CUDA (Required for Physics Simulation)
- **CUDA 11.8 or later** (required for Genesis physics engine)
- **GPU Memory**:
  - 8 GB minimum for basic simulations (single rover, small scenarios)
  - 16 GB recommended for interactive dashboard and small Monte Carlo experiments
  - 40 GB+ for large-scale RL training and parallel Monte Carlo studies

### System Dependencies
- CMake 3.20+ (for Genesis compilation)
- GCC/Clang with C++17 support
- pkg-config

## Installation Steps

### 1. Install CUDA 11.8+

Follow NVIDIA's official CUDA installation guide for your OS:
https://developer.nvidia.com/cuda-toolkit

**Verify CUDA installation:**
```bash
nvcc --version
nvidia-smi
```

Both commands should output version information without errors.

### 2. Install Genesis Physics Engine (from source)

The Genesis physics engine is required and must be compiled with GPU support.

```bash
# Clone Genesis repository
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis

# Install with GPU support (CUDA)
pip install -e .
```

Genesis compilation will link against your CUDA installation. If this step fails, verify CUDA 11.8+ is installed and in your PATH.

**Verify Genesis installation:**
```bash
python -c "import genesis; print(genesis.__version__)"
```

### 3. Clone Moon Rover Repository

```bash
git clone <moon-rover-repo-url>
cd Moon\ Rover
```

### 4. Install Moon Rover in Development Mode

Install the core package with optional feature groups as needed.

**Core installation (physics simulation only):**
```bash
pip install -e .
```

**Development installation (includes testing and documentation tools):**
```bash
pip install -e ".[dev]"
```

**Reinforcement Learning stack (imitation learning, policy training):**
```bash
pip install -e ".[rl]"
```

**ROS2 Hardware-in-the-Loop support:**
```bash
pip install -e ".[ros2]"
```

**All extras:**
```bash
pip install -e ".[dev,rl,ros2]"
```

### 5. Verify Installation

Test that all core components load correctly:

```bash
# Test Genesis and Moon Rover imports
python -c "from moon_rover.environment import MoonRoverEnv; print('✓ Moon Rover imports OK')"

# Run validation checks on all configs
moon-rover validate

# Display help for available commands
moon-rover --help
```

## Optional: Docker Setup

**Note:** Docker support is planned for future releases (TBD). For now, please use native installation.

## Troubleshooting

### "CUDA not found" Error
**Symptom:** Genesis compilation fails with `Could not find CUDA`

**Solution:**
1. Verify CUDA is installed: `nvcc --version`
2. Check CUDA is in PATH: `echo $CUDA_PATH` (should not be empty)
3. If PATH is missing, add to your shell profile:
   ```bash
   export CUDA_PATH=/usr/local/cuda-11.8  # Adjust path to your CUDA version
   export PATH=$CUDA_PATH/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
   ```
4. Re-run Genesis installation

### "Genesis import fails" Error
**Symptom:** `ImportError: cannot import name 'genesis'`

**Solution:**
1. Verify Genesis compiled successfully: `pip show genesis`
2. If not installed, re-run Genesis installation from source
3. Check GPU is visible: `python -c "import torch; print(torch.cuda.is_available())"`
4. If GPU not detected, verify NVIDIA drivers: `nvidia-smi`

### "Out of Memory" During Simulation
**Symptom:** `CUDA out of memory` error during `moon-rover run`

**Solution:**
1. Reduce simulation complexity: smaller terrain, fewer rovers
2. Disable visualization: use `--headless` flag (when available)
3. Reduce timestep or physics substeps in scenario YAML
4. For RL training, reduce batch size and worker count

### "URDF parsing fails" Error
**Symptom:** Validation fails with URDF errors

**Solution:**
1. Verify URDF files are in `src/moon_rover/assets/urdf/`
2. Check URDF syntax with: `moon-rover validate`
3. Review error message for specific URDF file and line number

## Sandbox Mode Limitations

**Note for cloud/browser environments:** If running in a sandboxed environment (e.g., browser-based notebooks or restricted containers):
- GPU hardware acceleration is **not available**
- Genesis physics engine will fall back to CPU mode (slow)
- Some advanced features may be disabled

For full fidelity simulations and RL training, work outside sandboxed environments with direct GPU access.

## Development Workflow

After installation, you can:

1. **Run a scenario:** `moon-rover run scenarios/v1_flat_single_rover.yaml`
2. **Launch dashboard:** `moon-rover dashboard --port 8080` (then visit http://localhost:8080)
3. **Run validation tests:** `pytest tests/`
4. **Edit and test configs:** Modify YAML files in `scenarios/` and re-run

For detailed usage, see [CLI Guide](./CLI.md) and [Configuration Reference](./CONFIG.md).
