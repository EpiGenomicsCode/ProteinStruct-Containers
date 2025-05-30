
"""Singularity launch script for Chai Lab Singularity image."""

import os
import sys
import pathlib
import signal
from typing import List, Tuple, Optional
import multiprocessing
import datetime

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

# --- Define the location of your Chai Lab Singularity image ---
# Option 1: Use an environment variable (recommended)
if 'CHAI_SIF' in os.environ:
    _CHAI_SIF_PATH = os.environ['CHAI_SIF']
# Option 2: Hardcode the path (replace with your actual path)
else:
    _CHAI_SIF_PATH = '/path/to/your/chai_lab.sif'  # PLEASE UPDATE THIS

# --- Temporary directory ---
if 'TMP' in os.environ:
    tmp_dir = os.environ['TMP']
elif 'TMPDIR' in os.environ:
    tmp_dir = os.environ['TMPDIR']
else:
    tmp_dir = '/tmp'

# Default path to a directory that will store the results.
output_dir_default = os.path.join(os.getcwd(), 'chailab_output')

#### END USER CONFIGURATION ####

_ROOT_MOUNT_DIRECTORY = '/mnt/'
FLAGS = flags.FLAGS

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
    else: # it's a file
        source_path = os.path.dirname(host_path)
        target_path = target_base # Mount the directory containing the file
        container_path = os.path.join(target_path, os.path.basename(host_path))


    # Create target directory on host if it doesn't exist for output/tmp
    # For input files, the source_path (directory part) must exist.
    if mount_point_name.startswith('output') or mount_point_name == 'tmp':
       os.makedirs(source_path, exist_ok=True)
    elif not os.path.exists(source_path):
        logging.error(f"Host path for binding does not exist: {source_path} (for mount {mount_point_name})")
        sys.exit(1)


    bind_spec = f'{source_path}:{target_path}'
    logging.info('Binding %s -> %s (container path: %s)', source_path, target_path, container_path)
    return (bind_spec, container_path)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # Check SIF path after flags are parsed, so it can be overridden
    sif_path = FLAGS.sif_path if FLAGS.sif_path else _CHAI_SIF_PATH
    if not os.path.exists(sif_path):
        print(f"Error: Singularity image not found at '{sif_path}'.")
        print("Please set the CHAI_SIF environment variable, update the _CHAI_SIF_PATH in this script, or use the --sif_path flag.")
        sys.exit(1)
    
    try:
        singularity_image = Client.load(sif_path)
    except Exception as e:
        print(f"Error loading Singularity image '{sif_path}': {e}")
        sys.exit(1)

    logging.info(f'INFO: Using Singularity image: {sif_path}')
    logging.info(f'INFO: Host temporary directory: {tmp_dir}')
    logging.info(f'INFO: Default base output directory: {output_dir_default}')


    # --- Argument validation ---
    if not FLAGS.fasta_file:
        raise app.UsageError('Missing required argument: --fasta_file')
    if not FLAGS.output_dir:
        raise app.UsageError('Missing required argument: --output_dir')


    # --- Prepare Singularity Bind Mounts ---
    binds = []
    chai_command_args = []

    # FASTA file
    bind_spec, container_fasta_file = _create_bind('fasta', FLAGS.fasta_file, is_dir=False)
    binds.append(bind_spec)
    # chai-lab fold expects fasta_file as a positional argument

    # Output directory
    # Ensure the output directory handling is correct, especially with force_output_dir
    actual_output_dir = FLAGS.output_dir
    if os.path.exists(actual_output_dir) and os.listdir(actual_output_dir) and not FLAGS.force_output_dir:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        actual_output_dir = os.path.join(actual_output_dir, f'run_{timestamp}')
        logging.warning(
            f"Output directory {FLAGS.output_dir} is not empty. "
            f"Using timestamped subdirectory: {actual_output_dir}"
        )
    os.makedirs(actual_output_dir, exist_ok=True)
    
    bind_spec, container_output_dir = _create_bind('output', actual_output_dir, is_dir=True)
    binds.append(bind_spec)
    # chai-lab fold expects output_dir as a positional argument


    # Optional path arguments that need binding
    if FLAGS.msa_directory:
        bind_spec, container_msa_dir = _create_bind('msa_dir', FLAGS.msa_directory, is_dir=True)
        binds.append(bind_spec)
        chai_command_args.append(f'--msa-directory={container_msa_dir}')

    if FLAGS.constraint_path:
        bind_spec, container_constraint_path = _create_bind('constraints', FLAGS.constraint_path, is_dir=False)
        binds.append(bind_spec)
        chai_command_args.append(f'--constraint-path={container_constraint_path}')

    if FLAGS.template_hits_path:
        bind_spec, container_template_hits = _create_bind('template_hits', FLAGS.template_hits_path, is_dir=False)
        binds.append(bind_spec)
        chai_command_args.append(f'--template-hits-path={container_template_hits}')
    
    # Temporary directory
    bind_spec, container_tmp_dir = _create_bind('tmp', tmp_dir, is_dir=True)
    binds.append(bind_spec)
    # Singularity typically inherits TMPDIR, but binding explicitly can be safer.
    # Chai might also use this.

    # --- Construct Chai Command ---
    # Base command: chai-lab fold <fasta_file> <output_dir>
    chai_exec_command = ['chai-lab', 'fold', container_fasta_file, container_output_dir]

    # Add boolean flags (only if True or if different from default where applicable)
    if FLAGS.use_esm_embeddings is False: # Default is True for chai-lab
        chai_command_args.append('--use-esm-embeddings=false')
    if FLAGS.use_msa_server: # Default is False
        chai_command_args.append('--use-msa-server')
    if FLAGS.use_templates_server: # Default is False
        chai_command_args.append('--use-templates-server')
    if FLAGS.low_memory is False: # Default is True for chai-lab
        chai_command_args.append('--low-memory=false')

    # Add flags with values
    if FLAGS.msa_server_url != "https://api.colabfold.com": # Default for chai-lab
        chai_command_args.append(f'--msa-server-url={FLAGS.msa_server_url}')
    if FLAGS.recycle_msa_subsample != 0: # Default for chai-lab
        chai_command_args.append(f'--recycle-msa-subsample={FLAGS.recycle_msa_subsample}')
    if FLAGS.num_trunk_recycles != 3: # Default for chai-lab
        chai_command_args.append(f'--num-trunk-recycles={FLAGS.num_trunk_recycles}')
    if FLAGS.num_diffn_timesteps != 200: # Default for chai-lab
        chai_command_args.append(f'--num-diffn-timesteps={FLAGS.num_diffn_timesteps}')
    if FLAGS.num_diffn_samples != 5: # Default for chai-lab
        chai_command_args.append(f'--num-diffn-samples={FLAGS.num_diffn_samples}')
    if FLAGS.num_trunk_samples != 1: # Default for chai-lab
        chai_command_args.append(f'--num-trunk-samples={FLAGS.num_trunk_samples}')
    if FLAGS.seed is not None:
        chai_command_args.append(f'--seed={FLAGS.seed}')
    if FLAGS.device: # chai-lab device flag
        chai_command_args.append(f'--device={FLAGS.device}')

    full_chai_command = chai_exec_command + chai_command_args

    # --- Prepare Singularity Options ---
    singularity_options = [
        '--bind', f'{",".join(binds)}',
        '--env', f'NVIDIA_VISIBLE_DEVICES={FLAGS.gpu_devices}',
        # Consider adding other env vars if Chai benefits from them, e.g.
        # '--env', 'XLA_PYTHON_CLIENT_MEM_FRACTION=0.9',
        # '--env', 'TF_FORCE_GPU_ALLOW_GROWTH=true',
    ]
    if container_tmp_dir: # Ensure TMPDIR is set inside container if we bind it
        singularity_options.extend(['--env', f'TMPDIR={container_tmp_dir}'])


    # --- Execute Singularity Command ---
    logging.info('Running Singularity command:')
    logging.info(f'Image: {sif_path}')
    logging.info(f'Singularity Options: {singularity_options}')
    logging.info(f'Chai Command: {" ".join(full_chai_command)}')

    try:
        result = Client.execute(
                 singularity_image,
                 full_chai_command,
                 nv=FLAGS.use_gpu,
                 options=singularity_options,
                 stream=True # Stream stdout/stderr
               )
        for line in result:
            print(line, end='') # Print container output in real-time

    except Exception as e:
        logging.error(f"Error executing Singularity command: {e}")
        # Attempt to clean up output dir if it was created by this script and is empty
        # This is a bit more complex if we created a timestamped one
        # For simplicity, just log the error. A more robust cleanup might be needed.
        sys.exit(1)

    logging.info('Chai Lab prediction finished.')


if __name__ == '__main__':
    # --- Define Command Line Flags ---
    flags.DEFINE_string(
        'sif_path', None,
        'Path to the Chai Lab Singularity image (.sif) file. '
        'Overrides CHAI_SIF environment variable and the hardcoded path.')
    flags.DEFINE_string(
        'fasta_file', None, 'Path to the input FASTA file (required).')
    flags.DEFINE_string(
        'output_dir', output_dir_default,
        'Path to a directory for storing results. A timestamped subdirectory '
        'may be created if the directory is non-empty and force_output_dir is False.')
    flags.DEFINE_boolean(
        'force_output_dir', False,
        'If True, use the exact output directory path even if it exists and is '
        'non-empty. Be careful, this may overwrite existing files. If False '
        '(default), a timestamped subdirectory is created if the target directory '
        'is non-empty.')
    
    # GPU Flags
    flags.DEFINE_boolean(
        'use_gpu', True, 'Enable NVIDIA runtime (--nv flag) to run with GPUs.')
    flags.DEFINE_string(
        'gpu_devices', 'all',
        'Comma separated list of GPU devices to pass to NVIDIA_VISIBLE_DEVICES '
        'environment variable inside the container (e.g., "0,1" or "all").')

    # Chai-lab fold specific flags
    flags.DEFINE_boolean(
        'use_esm_embeddings', True, '(Chai default: True) Whether to use ESM embeddings.')
    flags.DEFINE_boolean(
        'use_msa_server', False, '(Chai default: False) Whether to use the MSA server.')
    flags.DEFINE_string(
        'msa_server_url', 'https://api.colabfold.com', '(Chai default) URL of the MSA server.')
    flags.DEFINE_string(
        'msa_directory', None, 'Path to the directory containing MSAs.')
    flags.DEFINE_string(
        'constraint_path', None, 'Path to the constraints file.')
    flags.DEFINE_boolean(
        'use_templates_server', False, '(Chai default: False) Whether to use the templates server.')
    flags.DEFINE_string(
        'template_hits_path', None, 'Path to the template hits file.')
    flags.DEFINE_integer(
        'recycle_msa_subsample', 0, '(Chai default: 0) Number of MSA subsamples to recycle.')
    flags.DEFINE_integer(
        'num_trunk_recycles', 3, '(Chai default: 3) Number of trunk recycles.')
    flags.DEFINE_integer(
        'num_diffn_timesteps', 200, '(Chai default: 200) Number of diffusion timesteps.')
    flags.DEFINE_integer(
        'num_diffn_samples', 5, '(Chai default: 5) Number of diffusion samples.')
    flags.DEFINE_integer(
        'num_trunk_samples', 1, '(Chai default: 1) Number of trunk samples.')
    flags.DEFINE_integer(
        'seed', None, 'Random seed.')
    flags.DEFINE_string(
        'device', None, 'Device to use for chai-lab (e.g., "cuda:0"). '
        'If --use_gpu is True, Singularity will make GPUs available; '
        'this flag can specify which one for chai-lab if needed.')
    flags.DEFINE_boolean(
        'low_memory', True, '(Chai default: True) Whether to use low memory mode.')
    
    # Mark required flags
    flags.mark_flag_as_required('fasta_file')
    # output_dir has a default, so not strictly required from user but essential for script

    app.run(main) 