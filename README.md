# AlphaFold 3 Singularity Container Build

This repository contains scripts to build a Singularity container for AlphaFold 3, optimized for both **ARM64** and **x86** systems with NVIDIA GPUs. This setup has been tested on NVIDIA GPUs for both ARM and x86 based systems.

## Architecture Overview

![Architecture Diagram](Arch.png)

## Prerequisites

-   Access to a system with Singularity (or Apptainer) installed.
-   Access to the internet to download dependencies and clone repositories.
-   Access to a SLURM-managed cluster (if using the provided `build_af3.slurm` script).

## Database Download

AlphaFold 3 requires large sequence and structure databases to function.

**Requirement:** You must download these databases separately.
**Recommendation:** Use the official script provided by Google DeepMind:

1. Clone the official AlphaFold 3 repository:
   ```bash
   git clone https://github.com/google-deepmind/alphafold3.git
   cd alphafold3
   ```
2. Run the download script (this requires `wget` and `zstd`):
   ```bash
   ./fetch_databases.sh /path/to/your/database/storage
   ```
   Replace `/path/to/your/database/storage` with the desired location.

**Note:** The databases require significant disk space (~252 GB download, ~630 GB uncompressed). An SSD is recommended for better performance.

## Files

-   `build_af3.slurm`: SLURM batch script to build the Singularity container. **Requires user modification.**
-   `alphafold3_arm.def`: Singularity definition file optimized for ARM64 systems with NVIDIA GPUs (e.g., Grace Hopper).
-   `alphafold3_x86.def`: Singularity definition file for x86 systems with NVIDIA GPUs.
-   `run_alphafold3_launcher.py`: Python script for convenient execution of the container.

## Pre-built Container (Sylabs Cloud)

A pre-built Singularity image file (`.sif`) based on this definition is available on Sylabs Cloud:

*   **Link:** [Sylabs Cloud Link Placeholder - To Be Added]


## Building the Container

1.  **Configure SLURM Script**: Before submitting, open `build_af3.slurm` and replace the placeholder values:
    *   `#SBATCH --partition=YOUR_PARTITION`: Set this to the appropriate SLURM partition/queue for your cluster.
    *   `#SBATCH --account=YOUR_ACCOUNT`: Set this to your SLURM allocation/account name.
    *   *(Optional)* Adjust other SBATCH directives (like `--time`, `--mem`, `--cpus-per-task`, `--gres`) as needed for your environment and resource limits.

2.  **Submit the SLURM Job**: Use the `sbatch` command to submit the build job, specifying the target architecture (`arm` or `x86`) and the desired build directory.

    ```bash
    sbatch build_af3.slurm arm /path/to/your/build/directory
    ```
    
    Or for x86 systems:
    
    ```bash
    sbatch build_af3.slurm x86 /path/to/your/build/directory
    ```

3.  **Monitor the Build**: Check the output and error files (e.g., `alphafold3_build_*.out`, `alphafold3_build_*.err`, where `*` is the job ID) for progress and potential issues.

4.  **Locate the Container**: Upon successful completion, the container image (`alphafold3_arm.sif` or `alphafold3_x86.sif`, depending on your architecture choice) will be located in the build directory you specified when submitting the job.

## CUDA Version Compatibility

> ⚠️ **Important**: The definition files have the CUDA version hardcoded (currently set to CUDA 12.6.0).

This might cause compatibility issues if your host system uses a different CUDA version. You have two options:

1. **Modify the Definition File**: Edit the `From:` line in the definition file to match your system's CUDA version:
   ```
   Bootstrap: docker
   From: nvidia/cuda:XX.X.X-runtime-ubuntu22.04
   ```
   Replace `XX.X.X` with your system's CUDA version.

2. **Use Environment Variables**: Override default settings when running the container to prevent CUDA errors.

### System Requirements

- NVIDIA Driver version 525.60.13 or newer
- GPU with at least 8GB VRAM (16GB+ recommended)
- Singularity/Apptainer 3.8.0 or newer

### Advanced Configuration

The container automatically configures itself for the available GPU hardware. To override automatic settings or address CUDA compatibility issues, you can set environment variables when running:

```bash
singularity run --nv \
  -e XLA_CLIENT_MEM_FRACTION=0.85 \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e JAX_ENABLE_FLASH_ATTENTION=false \
  ... rest of command ...
```

### Troubleshooting

- If you encounter CUDA errors, try updating your NVIDIA drivers to the latest version
- For memory errors, decrease the `XLA_CLIENT_MEM_FRACTION` value (e.g., 0.75 or 0.5)
- Some systems may need to use `apptainer` instead of `singularity` command
- If you continue experiencing CUDA version mismatches, rebuilding the container with the matching CUDA version is recommended

## Running the Container

Refer to the `%help` section within the definition files (`alphafold3_arm.def` or `alphafold3_x86.def`) for architecture-specific instructions on running the built containers.

## Running Predictions (using Launcher Script)

The `run_alphafold3_launcher.py` script provides a convenient way to run predictions using the Singularity container.

1.  **Prerequisites:**
    *   Python 3.x
    *   `spython` and `absl-py` Python libraries: `pip install spython absl-py`
    *   A built Singularity container for your architecture (`alphafold3_arm.sif` or `alphafold3_x86.sif`).
    *   Downloaded AlphaFold 3 model parameters.
    *   Downloaded databases (see Database Download section).

2.  **Configuration:**
    *   Update the `_ALPHAFOLD3_SIF_PATH` variable inside `run_alphafold3_launcher.py` to point to your appropriate architecture-specific SIF file, OR set the `ALPHAFOLD3_SIF` environment variable.

3.  **Execution:**
    ```bash
    python run_alphafold3_launcher.py \
        --json_path=/path/to/input.json \
        --model_dir=/path/to/model_params \
        --db_dir=/path/to/databases \
        --output_dir=/path/to/output \
        [--other-flags...]
    ```
    Replace the example paths with your actual paths.

    **Key Flags:**
    *   `--json_path`: Path to a single input JSON file.
    *   `--input_dir`: Path to a directory of input JSON files (alternative to `--json_path`).
    *   `--model_dir`: Path to the downloaded AlphaFold 3 model parameters.
    *   `--db_dir`: Path(s) to the downloaded databases (can be specified multiple times).
    *   `--output_dir`: Directory where results will be saved.
    *   `--use_gpu`: Set to `false` to run without GPU (only data pipeline).
    *   `--run_data_pipeline=false`: Skip the data pipeline step.
    *   `--run_inference=false`: Skip the inference step.
    *   Run `python run_alphafold3_launcher.py --help` to see all available options.