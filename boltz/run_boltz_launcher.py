"""Singularity launch script for Boltz Singularity image."""

import os
import sys
import pathlib
import signal
from typing import List, Tuple, Optional
import multiprocessing
import datetime
import tempfile

from absl import app
from absl import flags
from absl import logging

try:
    from spython.main import Client
except ImportError:
    print("Error: spython library not found. Please install it: pip install spython")
    sys.exit(1)

#### USER CONFIGURATION ####

# --- Define the location of your Boltz Singularity image ---
# Option 1: Use an environment variable (recommended)
if 'BOLTZ_SIF' in os.environ:
    _BOLTZ_SIF_PATH = os.environ['BOLTZ_SIF']
# Option 2: Hardcode the path (replace with your actual path)
else:
    _BOLTZ_SIF_PATH = '/path/to/your/boltz.sif'  # PLEASE UPDATE THIS

# --- Default Boltz cache directory ---
_BOLTZ_CACHE_DEFAULT = os.path.expanduser('~/.boltz')
if 'BOLTZ_CACHE' in os.environ:
    _BOLTZ_CACHE_DEFAULT = os.environ['BOLTZ_CACHE']

# --- Temporary directory ---
if 'TMP' in os.environ:
    tmp_dir_host = os.environ['TMP']
elif 'TMPDIR' in os.environ:
    tmp_dir_host = os.environ['TMPDIR']
else:
    tmp_dir_host = '/tmp'

# Default path to a directory that will store the results.
output_dir_default = os.path.join(os.getcwd(), 'boltz_output')

#### END USER CONFIGURATION ####

_ROOT_MOUNT_DIRECTORY = '/mnt_launcher/' # Using a more unique root to avoid clashes
FLAGS = flags.FLAGS

def _create_bind(mount_point_name: str, host_path: str, is_dir: bool = True, read_only: bool = False) -> Tuple[str, str]:
    """Create a bind mount specification for Singularity.

    Args:
        mount_point_name: A descriptive name for the mount point (used for target path).
        host_path: The absolute path on the host system.
        is_dir: Whether the host_path is a directory.
        read_only: Whether to mount as read-only (appends ':ro').

    Returns:
        A tuple containing:
          - The bind string for Singularity ('host_path:target_path[:ro]').
          - The corresponding path inside the container.
    """
    host_path = os.path.abspath(os.path.expanduser(host_path))
    target_base = os.path.join(_ROOT_MOUNT_DIRECTORY, mount_point_name)

    if is_dir:
        source_path = host_path
        target_path_container = target_base
    else: # it's a file
        source_path = os.path.dirname(host_path)
        target_path_container = os.path.join(target_base, os.path.basename(host_path))
        # We bind the directory containing the file, and adjust container path
        # So, the actual mount target for singularity is target_base

    # Create target directory on host if it doesn't exist for output/tmp/cache
    if mount_point_name.startswith('output') or mount_point_name == 'tmp' or mount_point_name == 'boltz_cache':
       os.makedirs(host_path, exist_ok=True) # source_path is host_path for dirs
    elif not os.path.exists(source_path): # For other inputs like data or checkpoint file's dir
        logging.error(f"Host path for binding does not exist: {source_path} (for mount {mount_point_name})")
        sys.exit(1)
    
    # For file binds, the actual target in singularity is the directory.
    # The container_path variable correctly reflects the file path within that mounted dir.
    actual_bind_target = target_base 

    bind_spec = f'{source_path}:{actual_bind_target}'
    if read_only:
        bind_spec += ':ro'
    
    logging.info('Binding %s -> %s (container path for item: %s, read_only: %s)',
                 source_path, actual_bind_target, target_path_container, read_only)
    return (bind_spec, target_path_container)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    sif_path = FLAGS.sif_path if FLAGS.sif_path else _BOLTZ_SIF_PATH
    if not os.path.exists(sif_path):
        print(f"Error: Singularity image not found at '{sif_path}'.")
        print("Please set BOLTZ_SIF, update _BOLTZ_SIF_PATH, or use --sif_path.")
        sys.exit(1)
    
    try:
        singularity_image = Client.load(sif_path)
    except Exception as e:
        print(f"Error loading Singularity image '{sif_path}': {e}")
        sys.exit(1)

    logging.info(f'INFO: Using Singularity image: {sif_path}')
    logging.info(f'INFO: Host temporary directory: {tmp_dir_host}')
    logging.info(f'INFO: Default base output directory: {output_dir_default}')
    logging.info(f'INFO: Default Boltz cache directory: {_BOLTZ_CACHE_DEFAULT}')

    # --- Argument validation ---
    if not FLAGS.input_data:
        raise app.UsageError('Missing required argument: --input_data')

    # --- Prepare Singularity Bind Mounts ---
    binds = []
    boltz_args = []

    # Input Data (corresponds to DATA positional arg for boltz predict)
    # This can be a file or directory according to boltz help
    is_input_data_dir = os.path.isdir(FLAGS.input_data)
    bind_spec, container_input_data = _create_bind(
        'input_data', FLAGS.input_data, is_dir=is_input_data_dir, read_only=True
    )
    binds.append(bind_spec)
    # boltz_args will have this as the first positional argument after 'predict'

    # Output Directory (--out_dir for boltz)
    actual_output_dir = FLAGS.out_dir
    # Handle timestamped subdirectory for output if dir exists and is not empty
    # (Similar logic to other launchers can be added here if desired, but Boltz has --override)
    os.makedirs(actual_output_dir, exist_ok=True)
    bind_spec, container_output_dir = _create_bind('output', actual_output_dir, is_dir=True)
    binds.append(bind_spec)
    boltz_args.append(f'--out_dir={container_output_dir}')

    # Boltz Cache Directory (--cache for boltz)
    actual_boltz_cache_dir = FLAGS.boltz_cache_dir
    os.makedirs(actual_boltz_cache_dir, exist_ok=True)
    bind_spec, container_boltz_cache = _create_bind('boltz_cache', actual_boltz_cache_dir, is_dir=True)
    binds.append(bind_spec)
    boltz_args.append(f'--cache={container_boltz_cache}')

    # Checkpoint File (optional, --checkpoint for boltz)
    if FLAGS.checkpoint:
        if not os.path.isfile(FLAGS.checkpoint):
            logging.error(f"Checkpoint file not found: {FLAGS.checkpoint}")
            sys.exit(1)
        bind_spec, container_checkpoint = _create_bind('checkpoint_file', FLAGS.checkpoint, is_dir=False, read_only=True)
        binds.append(bind_spec)
        boltz_args.append(f'--checkpoint={container_checkpoint}')

    # Temporary directory for Singularity itself
    bind_spec, container_tmp_dir = _create_bind('tmp', tmp_dir_host, is_dir=True)
    binds.append(bind_spec)
    
    # --- Construct Boltz Command ---
    # Base command: boltz predict <DATA_container_path>
    boltz_exec_command = ['boltz', 'predict', container_input_data]

    # Add other optional flags based on their values
    # For flags with distinct --foo / --no-foo behavior:
    boltz_args.append('--write_full_pae' if FLAGS.write_full_pae else '--no-write-full-pae')
    boltz_args.append('--write_full_pde' if FLAGS.write_full_pde else '--no-write-full-pde')
    boltz_args.append('--override' if FLAGS.override else '--no-override')
    boltz_args.append('--use_msa_server' if FLAGS.use_msa_server else '--no-use-msa-server')
    boltz_args.append('--potentials' if FLAGS.enable_potentials else '--no_potentials') # as per investigation

    if FLAGS.devices != 1:
        boltz_args.append(f'--devices={FLAGS.devices}')
    if FLAGS.accelerator != 'gpu':
        boltz_args.append(f'--accelerator={FLAGS.accelerator}')
    if FLAGS.recycling_steps != 3:
        boltz_args.append(f'--recycling_steps={FLAGS.recycling_steps}')
    if FLAGS.sampling_steps != 200:
        boltz_args.append(f'--sampling_steps={FLAGS.sampling_steps}')
    if FLAGS.diffusion_samples != 1:
        boltz_args.append(f'--diffusion_samples={FLAGS.diffusion_samples}')
    if abs(FLAGS.step_scale - 1.638) > 1e-6: # Comparing floats
        boltz_args.append(f'--step_scale={FLAGS.step_scale}')
    if FLAGS.output_format != 'mmcif':
        boltz_args.append(f'--output_format={FLAGS.output_format}')
    if FLAGS.num_workers != 2:
        boltz_args.append(f'--num_workers={FLAGS.num_workers}')
    if FLAGS.seed is not None:
        boltz_args.append(f'--seed={FLAGS.seed}')
    
    if FLAGS.use_msa_server: # These are only relevant if use_msa_server is true
        if FLAGS.msa_server_url != "https://api.colabfold.com":
            boltz_args.append(f'--msa_server_url={FLAGS.msa_server_url}')
        if FLAGS.msa_pairing_strategy != "greedy":
            boltz_args.append(f'--msa_pairing_strategy={FLAGS.msa_pairing_strategy}')

    full_boltz_command = boltz_exec_command + boltz_args

    # --- Prepare Singularity Options ---
    singularity_options = [
        '--bind', f'{",".join(binds)}',
        '--env', f'NVIDIA_VISIBLE_DEVICES={FLAGS.gpu_devices}',
        # Pass BOLTZ_CACHE to container env, pointing to the mounted cache
        '--env', f'BOLTZ_CACHE={container_boltz_cache}',
    ]
    if container_tmp_dir: # Ensure TMPDIR is set inside container if we bind it
        singularity_options.extend(['--env', f'TMPDIR={container_tmp_dir}'])

    # --- Execute Singularity Command ---
    logging.info('Running Singularity command:')
    logging.info(f'Image: {sif_path}')
    logging.info(f'Singularity Options: {singularity_options}')
    logging.info(f'Boltz Command: {" ".join(full_boltz_command)}')

    try:
        result = Client.execute(
                 singularity_image,
                 full_boltz_command,
                 nv=FLAGS.use_gpu,
                 options=singularity_options,
                 stream=True # Stream stdout/stderr
               )
        for line in result:
            print(line, end='')
    except Exception as e:
        logging.error(f"Error executing Singularity command: {e}")
        sys.exit(1)

    logging.info('Boltz prediction finished.')


if __name__ == '__main__':
    # --- Define Command Line Flags ---
    flags.DEFINE_string(
        'sif_path', None,
        'Path to the Boltz Singularity image (.sif) file. '
        'Overrides BOLTZ_SIF environment variable and the hardcoded path.')
    flags.DEFINE_string(
        'input_data', None, '(Required) Input data file or directory (FASTA/YAML). Corresponds to DATA positional arg for boltz.')
    flags.DEFINE_string(
        'out_dir', output_dir_default,
        'Output directory for predictions. Corresponds to --out_dir for boltz.')
    flags.DEFINE_string(
        'boltz_cache_dir', _BOLTZ_CACHE_DEFAULT,
        'Directory for Boltz to download data/models. Corresponds to --cache for boltz. '
        'Defaults to $BOLTZ_CACHE env var or ~/.boltz')
    flags.DEFINE_string(
        'checkpoint', None, 'Optional path to a model checkpoint file. Corresponds to --checkpoint for boltz.')

    # GPU/Accelerator Flags for Singularity and Boltz
    flags.DEFINE_boolean(
        'use_gpu', True, 'Enable NVIDIA runtime (--nv flag for Singularity) to run with GPUs.')
    flags.DEFINE_string(
        'gpu_devices', 'all',
        'Comma separated list of GPU devices for NVIDIA_VISIBLE_DEVICES (e.g., "0,1").')
    flags.DEFINE_integer(
        'devices', 1, 'Number of devices (e.g., GPUs) for Boltz prediction (--devices).')
    flags.DEFINE_enum(
        'accelerator', 'gpu', ['gpu', 'cpu', 'tpu'], 
        'Accelerator type for Boltz prediction (--accelerator).')

    # Boltz predict specific flags (as per investigation)
    flags.DEFINE_integer(
        'recycling_steps', 3, 'Number of recycling steps (--recycling_steps).')
    flags.DEFINE_integer(
        'sampling_steps', 200, 'Number of sampling steps (--sampling_steps).')
    flags.DEFINE_integer(
        'diffusion_samples', 1, 'Number of diffusion samples (--diffusion_samples).')
    flags.DEFINE_float(
        'step_scale', 1.638, 'Step scale for diffusion (--step_scale).')
    
    flags.DEFINE_boolean(
        'write_full_pae', False, 'Dump full PAE matrix? Appends --write_full_pae or --no-write-full-pae.')
    flags.DEFINE_boolean(
        'write_full_pde', False, 'Dump full PDE matrix? Appends --write_full_pde or --no-write-full_pde.')
    flags.DEFINE_enum(
        'output_format', 'mmcif', ['pdb', 'mmcif'], 'Output format for structures (--output_format).')
    flags.DEFINE_integer(
        'num_workers', 2, 'Number of data loader workers (--num_workers).')
    flags.DEFINE_boolean(
        'override', False, 'Override existing predictions? Appends --override or --no-override.')
    flags.DEFINE_integer(
        'seed', None, 'Random seed for reproducibility (--seed).')

    # MSA Server related flags for Boltz
    flags.DEFINE_boolean(
        'use_msa_server', False, 'Use MMSeqs2 server for MSA? Appends --use_msa_server or --no-use-msa-server.')
    flags.DEFINE_string(
        'msa_server_url', 'https://api.colabfold.com', 'MSA server URL (--msa_server_url).')
    flags.DEFINE_enum(
        'msa_pairing_strategy', 'greedy', ['greedy', 'complete'], 'MSA pairing strategy (--msa_pairing_strategy).')

    # Potentials flag for Boltz
    flags.DEFINE_boolean(
        'enable_potentials', True, 'Use potentials for steering? Appends --potentials or --no_potentials.')

    # Mark required flags
    flags.mark_flag_as_required('input_data')

    app.run(main) 