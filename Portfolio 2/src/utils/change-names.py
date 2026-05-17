import wandb

api = wandb.Api()
runs = api.runs("hhs-autonomous-systems/trackmania-rl")

counter = 1
for run in runs:
    if run.name.startswith("sweep"):  # only rename the ones with long names
        run.name = f"phase-1-sweep-{counter}"
        run.update()
        print(f"Renamed to: phase-1-sweep-{counter}")
        counter += 1