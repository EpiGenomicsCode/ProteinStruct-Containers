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
    echo "Error: Please provide the Singularity definition file path as the first argument"
    echo "       and the desired build directory path as the second argument."
    echo "Usage: sbatch $0 <definition_file> <build_directory>"
    exit 1
fi
DEFINITION_FILE=$1
BUILD_DIR=$2

# Set and create the build directory
mkdir -p "${BUILD_DIR}"

# Definition file (passed as $1) is assumed to exist relative to the submission directory or as an absolute path

# Build the Singularity container
cd "${BUILD_DIR}"
echo "Building container using definition file: ${DEFINITION_FILE}"
echo "Output container (alphafold3.sif) will be placed in: ${BUILD_DIR}"
singularity build --force alphafold3.sif "${DEFINITION_FILE}" 