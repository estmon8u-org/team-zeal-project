"""
Profiler script for the training process.

This script runs a short training session using cProfile to identify
performance bottlenecks in the training pipeline.

How to run:
1. Ensure your environment is set up and dependencies are installed.
2. From the project root directory, execute:
   python scripts/profile_training.py

How to analyze results:
After running the script, a 'training_profile.prof' file will be generated.
You can analyze this file using the pstats module in Python:

import pstats
from pstats import SortKey

# Load the profiling stats
p = pstats.Stats('training_profile.prof')

# Sort by cumulative time and print top 20 functions
print("--- Top 20 functions by cumulative time ---")
p.sort_stats(SortKey.CUMULATIVE).print_stats(20)

# Sort by total time spent in function and print top 20 functions
print("\n--- Top 20 functions by total time (tottime) ---")
p.sort_stats(SortKey.TIME).print_stats(20)

# If you want to see which functions called a specific function, e.g., 'forward':
# p.print_callers(.5, 'forward')

# If you want to see which functions were called by a specific function:
# p.print_callees(.5, 'forward')
"""
import cProfile
import pstats
import os
import sys

# To make hydra work when called from a script in a subdirectory,
# we need to adjust the path to let it find the main config directory.
# This assumes the script is in 'scripts/' and 'conf/' is at the project root.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Attempt to import the main training function
try:
    from drift_detector_pipeline.modeling.train import train as hydra_train_entry_point
except ImportError as e:
    print(f"Error importing training function: {e}")
    print("Please ensure that drift_detector_pipeline.modeling.train.train exists and is runnable.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)

def profile_training_run():
    """
    Wrapper function to execute a short training run for profiling.
    Uses Hydra's command-line override capabilities.
    """
    print("Starting profiled training run...")
    print("Note: This will run a very short training session (e.g., 1 epoch, limited batches).")

    # Store original command line args
    original_argv = sys.argv.copy()

    # Override Hydra configurations for a short run.
    # Adjust these as necessary for your project's specific Hydra config.
    # Common overrides include epochs, batch size, or specific dataset parameters
    # to limit the data processed.
    # Example: Force 1 epoch, and potentially a smaller batch size if your config supports it.
    # If your 'train' function is decorated with @hydra.main(config_path="../conf", config_name="config")
    # these overrides will be picked up.
    #
    # The path to `config_path` in `@hydra.main` inside `train.py` is relative to `train.py`.
    # Since we are running from `scripts/profile_training.py`, if `train.py` uses `config_path="../conf"`,
    # this should still work correctly as Hydra resolves paths relative to the main script it decorates.
    sys.argv = [
        sys.argv[0],  # Script name
        "training.epochs=1",
        "experiment.name=profiling_run",
        # Add other relevant overrides here.
        # For example, if you have a way to limit batches per epoch:
        # "datamodule.train_dataloader_cfg.limit_batches_per_epoch=10",
        # Or if your dataset module has a "fast_dev_run" or "debug" flag:
        # "datamodule.fast_dev_run=true"
        # Ensure these overrides match your actual Hydra configuration structure.
        # If your primary config is in `conf/config.yaml`, these will override values there.
    ]

    try:
        # The `hydra_train_entry_point` is the function decorated by @hydra.main.
        # Calling it will make Hydra parse the (now modified) sys.argv.
        hydra_train_entry_point()
    except Exception as e:
        print(f"An error occurred during the profiled training run: {e}")
        print("Please check your Hydra configuration and training script.")
        # Ensure to restore sys.argv in case of error too
        sys.argv = original_argv
        sys.exit(1)
    finally:
        # Restore original command line args
        sys.argv = original_argv

    print("Profiled training run finished.")

if __name__ == "__main__":
    profile_file = "training_profile.prof"

    print(f"Running profiler. Output will be saved to {profile_file}")

    # Run the profiler
    cProfile.runctx("profile_training_run()", globals(), locals(), filename=profile_file)

    print(f"\nProfiling complete. Stats saved to {profile_file}")
    print("To analyze the results, use pstats as described in the script's docstring.")

    # Example of printing basic stats directly after running
    print("\n--- Basic Stats (Top 5 by cumulative time) ---")
    with open("profiling_output.txt", "w") as f_out: # Save to a file as well
        p = pstats.Stats(profile_file, stream=f_out)
        p.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(10)
    
    p = pstats.Stats(profile_file) # Print to console
    p.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(5)

    print(f"\nMore detailed analysis can be performed using pstats interactively.")
    print(f"A summary has also been written to profiling_output.txt")
