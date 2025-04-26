#!/bin/bash
#SBATCH --job-name=alphafold3_build        # Descriptive job name
#SBATCH --output=alphafold3_build_%j.out   # Standard output file (%j expands to job ID)
#SBATCH --error=alphafold3_build_%j.err    # Standard error file (%j expands to job ID)
#SBATCH --partition=YOUR_PARTITION       # Specify the partition/queue name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --mem=256G
#SBATCH --account=YOUR_ACCOUNT         # Specify your allocation account
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --gpus-per-task=1

# Load necessary modules (uncomment if needed)
# module load singularity

# Check if arguments are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Please provide the architecture (arm or x86) as the first argument"
    echo "       and the desired build directory path as the second argument."
    echo "Usage: sbatch $0 <architecture> <build_directory>"
    echo "       where <architecture> is either 'arm' or 'x86'"
    exit 1
fi

ARCH=$1
BUILD_DIR=$2

# Determine the appropriate definition file based on architecture
if [ "${ARCH}" = "arm" ]; then
    DEFINITION_FILE="/Users/vinay/AF3-Container/alphafold3_arm.def"
    echo "Using ARM64 definition file: ${DEFINITION_FILE}"
elif [ "${ARCH}" = "x86" ]; then
    DEFINITION_FILE="/Users/vinay/AF3-Container/alphafold3_x86.def"
    echo "Using x86 definition file: ${DEFINITION_FILE}"
else
    echo "Error: Architecture must be either 'arm' or 'x86'"
    exit 1
fi

# Set and create the build directory
mkdir -p "${BUILD_DIR}"

# Build the Singularity container
cd "${BUILD_DIR}"
echo "Building container using definition file: ${DEFINITION_FILE}"
echo "Output container (alphafold3_${ARCH}.sif) will be placed in: ${BUILD_DIR}"
singularity build --force alphafold3_${ARCH}.sif "${DEFINITION_FILE}" 