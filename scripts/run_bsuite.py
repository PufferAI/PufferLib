import concurrent.futures
import subprocess

def execute_command(command):
    # Use subprocess to execute the command, replace with your execution logic
    result = subprocess.run(command, shell=True, capture_output=True)
    return result.stdout

from bsuite import sweep
commands = [
    f'python demo.py --train --track --env bsuite --env-kwargs.name {env}'
    for env in sweep.SWEEP
]

with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    future_to_command = {executor.submit(execute_command, cmd): cmd for cmd in commands}

    for future in concurrent.futures.as_completed(future_to_command):
        cmd = future_to_command[future]
        try:
            data = future.result()
            print(f"{cmd} output: {data}")
        except Exception as exc:
            print(f"{cmd} generated an exception: {exc}")
