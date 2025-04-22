import wandb
import pandas as pd

# Define sweep and project details
PROJECT_NAME = "jsuarez/pufferlib"
SWEEP_ID = "vdkhpxm9"

# Initialize API
api = wandb.Api()

# Fetch the sweep
sweep = api.sweep(f"{PROJECT_NAME}/{SWEEP_ID}")

# Collect all runs in the sweep
runs = sweep.runs
breakpoint()

# Gather data from all runs
data = []
for run in runs:
    # Get the run's history as a pandas DataFrame
    history = run.history()
    history["run_id"] = run.id  # Add run ID to differentiate runs
    data.append(history)

# Combine all run histories into a single DataFrame
combined_data = pd.concat(data, ignore_index=True)

# Save as CSV
combined_data.to_csv("sweep_data.csv", index=False)

# Save as JSON
combined_data.to_json("sweep_data.json", orient="records")

print("Sweep data exported to sweep_data.csv and sweep_data.json")

