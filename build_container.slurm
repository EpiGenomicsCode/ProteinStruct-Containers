#!/bin/bash
#SBATCH --job-name=${MODEL_DIR_NAME}_${ARCH}_build # Descriptive job name - NOTE: SLURM may not expand these script variables here
#SBATCH --output=${MODEL_DIR_NAME}_${ARCH}_build_%j.out   # Standard output file - NOTE: SLURM may not expand these script variables here
#SBATCH --error=${MODEL_DIR_NAME}_${ARCH}_build_%j.err    # Standard error file - NOTE: SLURM may not expand these script variables here
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
    echo "Error: Please provide the model directory name, architecture, and build directory."
    echo "Usage: sbatch $0 <model_directory_name> <architecture> <build_directory>"
    echo "       where <model_directory_name> is 'alphafold3', 'boltz', or 'chai_1'"
    echo "       and <architecture> is either 'arm' or 'x86'"
    exit 1
fi

MODEL_DIR_NAME=$1
ARCH=$2
BUILD_DIR=$3

# Determine the definition file name prefix based on the model directory name
DEF_FILE_PREFIX=""
if [ "${MODEL_DIR_NAME}" = "chai_1" ]; then
    DEF_FILE_PREFIX="chai_lab"
elif [ "${MODEL_DIR_NAME}" = "alphafold3" ]; then
    DEF_FILE_PREFIX="alphafold3"
elif [ "${MODEL_DIR_NAME}" = "boltz" ]; then
    DEF_FILE_PREFIX="boltz"
else
    echo "Error: Invalid model directory name '${MODEL_DIR_NAME}'. Must be 'alphafold3', 'boltz', or 'chai_1'."
    exit 1
fi

# Construct the path to the definition file
# Assumes this script is in the root, and .def files are in ../<model_dir_name>/<def_file_prefix>_<arch>.def
DEFINITION_FILE_PATH="../${MODEL_DIR_NAME}/${DEF_FILE_PREFIX}_${ARCH}.def"

echo "Using definition file: ${DEFINITION_FILE_PATH}"

# Validate architecture
if [ "${ARCH}" != "arm" ] && [ "${ARCH}" != "x86" ]; then
    echo "Error: Architecture must be either 'arm' or 'x86'"
    exit 1
fi

# Check if the definition file exists (relative to the script's assumed execution path before cd)
# This check is done from the perspective of where sbatch is run (assumed to be repo root)
ACTUAL_DEF_FILE_FOR_CHECK="${MODEL_DIR_NAME}/${DEF_FILE_PREFIX}_${ARCH}.def"
if [ ! -f "${ACTUAL_DEF_FILE_FOR_CHECK}" ]; then
    echo "Error: Definition file ${ACTUAL_DEF_FILE_FOR_CHECK} not found from repository root."
    echo "Please ensure the file exists in the correct model subdirectory."
    exit 1
fi

# Set and create the build directory
mkdir -p "${BUILD_DIR}"

# Build the Singularity container
cd "${BUILD_DIR}"
SIF_NAME="${MODEL_DIR_NAME}_${ARCH}.sif" # SIF name based on model directory and arch

echo "Building container ${SIF_NAME} using definition file: ${DEFINITION_FILE_PATH}"
echo "Output container will be placed in: $(pwd)"

singularity build --force "${SIF_NAME}" "${DEFINITION_FILE_PATH}"

if [ $? -eq 0 ]; then
    echo "Successfully built ${SIF_NAME} in $(pwd)"
else
    echo "Error building ${SIF_NAME}. Check logs in $(pwd)"
    exit 1
fi
