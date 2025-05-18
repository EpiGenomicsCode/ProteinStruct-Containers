#!/bin/bash
#SBATCH --job-name={MODEL_NAME}_${ARCH}_build # Descriptive job name
#SBATCH --output={MODEL_NAME}_${ARCH}_build_%j.out   # Standard output file (%j expands to job ID)
#SBATCH --error={MODEL_NAME}_${ARCH}_build_%j.err    # Standard error file (%j expands to job ID)
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
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Error: Please provide the model name, architecture, and build directory."
    echo "Usage: sbatch $0 <model_name> <architecture> <build_directory>"
    echo "       where <model_name> is e.g., 'alphafold3', 'boltz', 'chai_lab'"
    echo "       and <architecture> is either 'arm' or 'x86'"
    exit 1
fi

MODEL_NAME=$1
ARCH=$2
BUILD_DIR=$3

# Construct the definition file name
DEFINITION_FILE="${MODEL_NAME}_${ARCH}.def" # Assumes .def files are in the CWD or accessible path

echo "Using definition file: ${DEFINITION_FILE}"

# Validate architecture (optional, as file existence check below is primary)
if [ "${ARCH}" != "arm" ] && [ "${ARCH}" != "x86" ]; then
    echo "Error: Architecture must be either 'arm' or 'x86'"
    exit 1
fi

# Check if the definition file exists
if [ ! -f "${DEFINITION_FILE}" ]; then
    echo "Error: Definition file ${DEFINITION_FILE} not found."
    echo "Please ensure the file exists in the current directory or provide a correct path."
    exit 1
fi

# Set and create the build directory
mkdir -p "${BUILD_DIR}"

# Build the Singularity container
cd "${BUILD_DIR}"
SIF_NAME="${MODEL_NAME}_${ARCH}.sif"
echo "Building container using definition file: ${DEFINITION_FILE}"
echo "Output container (${SIF_NAME}) will be placed in: ${BUILD_DIR}"
singularity build --force "${SIF_NAME}" "../${DEFINITION_FILE}"
