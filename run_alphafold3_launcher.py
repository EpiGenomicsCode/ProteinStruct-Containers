#!/usr/bin/env python3
# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications Copyright 2024 Vinay Mathew.
# Adapted for AlphaFold 3 and Singularity execution based on AF2 example.

"""Singularity launch script for AlphaFold 3 Singularity image."""

import os
import sys
import pathlib
import signal
from typing import List, Tuple, Optional
import multiprocessing

from absl import app
from absl import flags
from absl import logging
try:
    from spython.main import Client
except ImportError:
    print("Error: spython library not found. Please install it: pip install spython")
    sys.exit(1)

import tempfile

#### USER CONFIGURATION ####

# --- Define the location of your AlphaFold 3 Singularity image ---
# Option 1: Use an environment variable (recommended)
if 'ALPHAFOLD3_SIF' in os.environ:
    _ALPHAFOLD3_SIF_PATH = os.environ['ALPHAFOLD3_SIF']
# Option 2: Hardcode the path (replace with your actual path)
else:
    _ALPHAFOLD3_SIF_PATH = '/path/to/your/alphafold3.sif'  # PLEASE UPDATE THIS

# Check if the configured SIF path exists
if not os.path.exists(_ALPHAFOLD3_SIF_PATH):
    print(f"Error: Singularity image not found at '{_ALPHAFOLD3_SIF_PATH}'.")
    print("Please set the ALPHAFOLD3_SIF environment variable or update the _ALPHAFOLD3_SIF_PATH in this script.")
    sys.exit(1)

try:
    singularity_image = Client.load(_ALPHAFOLD3_SIF_PATH)
except Exception as e:
    print(f"Error loading Singularity image '{_ALPHAFOLD3_SIF_PATH}': {e}")
    sys.exit(1)

# --- Temporary directory ---
if 'TMP' in os.environ:
    tmp_dir = os.environ['TMP']
elif 'TMPDIR' in os.environ:
    tmp_dir = os.environ['TMPDIR']
else:
    tmp_dir = '/tmp'

# Default path to a directory that will store the results.
# A subdirectory will be created within this for each run based on input name.
output_dir_default = os.path.join(os.getcwd(), 'alphafold3_output')

logging.info(f'INFO: Using Singularity image: {_ALPHAFOLD3_SIF_PATH}')
logging.info(f'INFO: Host temporary directory: {tmp_dir}')
logging.info(f'INFO: Default base output directory: {output_dir_default}')

#### END USER CONFIGURATION ####

# --- Define Command Line Flags ---
flags.DEFINE_string(
    'json_path', None,
    'Path to a single JSON file containing the prediction input specification. '
    'See AlphaFold 3 documentation for format details.')
flags.DEFINE_string(
    'input_dir', None,
    'Path to a directory containing multiple JSON input files. If specified, '
    '--json_path is ignored. Each JSON file will be processed.')
flags.DEFINE_string(
    'output_dir', output_dir_default,
    'Path to a base directory for storing results. A subdirectory will be '
    'created within this directory for each input JSON file processed, named '
    'after the "name" field in the JSON.')
flags.DEFINE_boolean(
    'force_output_dir', False,
    'If True, use the exact output directory path even if it exists and is '
    'non-empty. Be careful, this may overwrite existing files. If False '
    '(default), a timestamped subdirectory is created if the target directory '
    'is non-empty.')
flags.DEFINE_string(
    'model_dir', None,
    'Path to the directory containing the AlphaFold 3 model parameters.')
flags.DEFINE_list(
    'db_dir', None,
    'Paths to directories containing the genetic databases. Can be specified '
    'multiple times (e.g., for SSD fallback). The order matters: '
    'databases found in earlier directories are used preferentially. '
    'Example: --db_dir=/path/to/ssd/dbs --db_dir=/path/to/hdd/dbs')
flags.DEFINE_boolean(
    'use_gpu', True, 'Enable NVIDIA runtime (--nv flag) to run with GPUs.')
flags.DEFINE_string(
    'gpu_devices', 'all',
    'Comma separated list of GPU devices to pass to NVIDIA_VISIBLE_DEVICES '
    'environment variable inside the container.')
flags.DEFINE_boolean(
    'run_data_pipeline', True,
    'Run the data pipeline (genetic search, template search). Set to false '
    'if MSAs/templates are precomputed or provided in the input JSON.')
flags.DEFINE_boolean(
    'run_inference', True,
    'Run the inference pipeline (requires GPU if use_gpu=True). Set to false '
    'to only run the data pipeline.')
flags.DEFINE_integer(
    'jackhmmer_n_cpu', min(multiprocessing.cpu_count(), 8),
    'Number of CPUs to use for Jackhmmer inside the container.')
flags.DEFINE_integer(
    'nhmmer_n_cpu', min(multiprocessing.cpu_count(), 8),
    'Number of CPUs to use for Nhmmer inside the container.')
flags.DEFINE_string(
    'max_template_date', '2021-09-30',
    'Maximum template release date to consider (Format: YYYY-MM-DD). Also affects '
    'fallback behavior for ligand conformer generation.')
flags.DEFINE_integer(
    'conformer_max_iterations', 10000,
    'Maximum number of iterations for RDKit conformer generation for ligands '
    'specified via SMILES. Increase if experiencing conformer generation '
    'failures for complex ligands.')
flags.DEFINE_integer(
    'num_recycles', 10,
    'Number of recycles to use during inference.', lower_bound=1)
flags.DEFINE_integer(
    'num_diffusion_samples', 5,
    'Number of diffusion samples to generate per seed.', lower_bound=1)
flags.DEFINE_integer(
    'num_seeds', None,
    'If set, overrides the seeds in the JSON file. Only a single seed must be '
    'provided in the JSON, and this flag specifies how many total seeds '
    '(starting from the one in the JSON) should be run.', lower_bound=1)
flags.DEFINE_enum(
    'flash_attention_implementation',
    default='triton',
    enum_values=['triton', 'cudnn', 'xla'],
    help='Flash attention implementation to use inside the container (triton/cudnn require Ampere+ GPU).')
flags.DEFINE_boolean(
    'save_embeddings', False,
    'Whether to save the final trunk single and pair embeddings in the output.')

FLAGS = flags.FLAGS

_ROOT_MOUNT_DIRECTORY = '/mnt/'


def _create_bind(mount_point_name: str, host_path: str, is_dir: bool = True) -> Tuple[str, str]:
    """Create a bind mount specification for Singularity.

    Args:
        mount_point_name: A descriptive name for the mount point (used for target path).
        host_path: The absolute path on the host system.
        is_dir: Whether the host_path is a directory.

    Returns:
        A tuple containing:
          - The bind string for Singularity ('host_path:target_path').
          - The corresponding path inside the container.
    """
    host_path = os.path.abspath(host_path)
    target_base = os.path.join(_ROOT_MOUNT_DIRECTORY, mount_point_name)

    if is_dir:
        source_path = host_path
        target_path = target_base
        container_path = target_path
    else:
        source_path = os.path.dirname(host_path)
        target_path = target_base
        container_path = os.path.join(target_path, os.path.basename(host_path))

    # Create target directory on host if it doesn't exist for output/tmp
    if mount_point_name.startswith('output') or mount_point_name == 'tmp':
       os.makedirs(source_path, exist_ok=True)

    bind_spec = f'{source_path}:{target_path}'
    logging.info('Binding %s -> %s', source_path, target_path)
    return (bind_spec, container_path)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    if not FLAGS.json_path and not FLAGS.input_dir:
        raise app.UsageError('Either --json_path or --input_dir must be specified.')
    if FLAGS.json_path and FLAGS.input_dir:
        logging.warning('--input_dir specified, ignoring --json_path.')
    if not FLAGS.model_dir:
        raise app.UsageError('--model_dir must be specified.')
    if not FLAGS.db_dir:
        raise app.UsageError('--db_dir must be specified (can be provided multiple times).')

    # --- Prepare Singularity Bind Mounts ---
    binds = []
    command_args = [] # Arguments for the internal run_alphafold.py

    # Input JSON path or directory
    container_json_path = None
    container_input_dir = None
    if FLAGS.input_dir:
        bind_spec, container_input_dir = _create_bind('input_dir', FLAGS.input_dir, is_dir=True)
        binds.append(bind_spec)
        command_args.append(f'--input_dir={container_input_dir}')
    elif FLAGS.json_path:
        bind_spec, container_json_path = _create_bind('json_input', FLAGS.json_path, is_dir=False)
        binds.append(bind_spec)
        command_args.append(f'--json_path={container_json_path}')

    # Output directory
    bind_spec, container_output_dir = _create_bind('output', FLAGS.output_dir, is_dir=True)
    binds.append(bind_spec)
    command_args.append(f'--output_dir={container_output_dir}')

    # Model parameters directory
    bind_spec, container_model_dir = _create_bind('models', FLAGS.model_dir, is_dir=True)
    binds.append(bind_spec)
    command_args.append(f'--model_dir={container_model_dir}')

    # Database directories (potentially multiple)
    container_db_dirs = []
    for i, db_path in enumerate(FLAGS.db_dir):
        bind_spec, container_db_path = _create_bind(f'db_{i}', db_path, is_dir=True)
        binds.append(bind_spec)
        container_db_dirs.append(container_db_path)
        command_args.append(f'--db_dir={container_db_path}') # Pass each one

    # Temporary directory
    bind_spec, container_tmp_dir = _create_bind('tmp', tmp_dir, is_dir=True)
    binds.append(bind_spec)
    # Singularity typically inherits TMPDIR, but binding explicitly can be safer

    # --- Construct Command ---
    command_args.extend([
        f'--run_data_pipeline={str(FLAGS.run_data_pipeline).lower()}',
        f'--run_inference={str(FLAGS.run_inference).lower()}',
        f'--conformer_max_iterations={FLAGS.conformer_max_iterations}',
        # Pass through newly added flags
        f'--jackhmmer_n_cpu={FLAGS.jackhmmer_n_cpu}',
        f'--nhmmer_n_cpu={FLAGS.nhmmer_n_cpu}',
        f'--max_template_date={FLAGS.max_template_date}',
        f'--num_recycles={FLAGS.num_recycles}',
        f'--num_diffusion_samples={FLAGS.num_diffusion_samples}',
        f'--flash_attention_implementation={FLAGS.flash_attention_implementation}',
        f'--save_embeddings={str(FLAGS.save_embeddings).lower()}',
        f'--force_output_dir={str(FLAGS.force_output_dir).lower()}',
        # Useful for debugging within the container
        '--logtostderr',
    ])

    # Conditionally add num_seeds if specified by the user
    if FLAGS.num_seeds is not None:
        command_args.append(f'--num_seeds={FLAGS.num_seeds}')

    # Prepend the python execution command
    run_script_path = '/app/run_alphafold.py' # Assuming this is the path inside the SIF
    full_command = ['python', run_script_path] + command_args

    # --- Prepare Singularity Options ---
    options = [
        '--bind', f'{",".join(binds)}',
        '--env', f'NVIDIA_VISIBLE_DEVICES={FLAGS.gpu_devices}',
        # Add performance-related env vars if needed (might depend on GPU/setup)
        # '--env', 'TF_FORCE_UNIFIED_MEMORY=1',
        # '--env', 'XLA_PYTHON_CLIENT_MEM_FRACTION=4.0', # Adjust as needed
    ]

    # --- Execute Singularity Command ---
    logging.info('Running Singularity command:')
    logging.info(f'Image: {_ALPHAFOLD3_SIF_PATH}')
    logging.info(f'Options: {options}')
    logging.info(f'Command: {" ".join(full_command)}')

    try:
        # Use Client.execute which streams output, better for long jobs
        # Client.run returns all output only at the end.
        result = Client.execute(
                 singularity_image,
                 full_command,
                 nv=FLAGS.use_gpu,
                 options=options,
                 stream=True # Stream stdout/stderr
               )
        for line in result:
            print(line, end='') # Print container output in real-time

    except Exception as e:
        logging.error(f"Error executing Singularity command: {e}")
        # Attempt to clean up default output dir if it was created and is empty
        try:
            if FLAGS.output_dir == output_dir_default and os.path.exists(output_dir_default) and not os.listdir(output_dir_default):
                logging.info(f"Attempting to remove empty default output directory: {output_dir_default}")
                os.rmdir(output_dir_default)
        except OSError as rm_err:
            logging.warning(f"Could not remove default output directory: {rm_err}")
        sys.exit(1)

    logging.info('AlphaFold 3 prediction finished.')


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'model_dir',
        'db_dir',
        # Either json_path or input_dir is required, checked in main()
    ])
    app.run(main) 