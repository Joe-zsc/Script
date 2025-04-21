import os, sys
from pathlib import Path
import json
import random

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
sys.path.append(curr_path)  # add current terminal path to sys.path
from util import UTIL, console, set_seed

scenario_path = UTIL.project_path / "scenarios"


def create_chain_scenarios(source_scenario_path: Path,
                           chain_length=1,
                           seed=0,
                           saved_path=None):
    set_seed(seed)
    saved_path = (saved_path if saved_path else UTIL.project_path /
                  "scenarios" / "chain")
    saved_path.mkdir(parents=True, exist_ok=True)
    if source_scenario_path.is_dir():
        scenario_list = [path for path in source_scenario_path.iterdir()]
    else:
        scenario_list = [source_scenario_path]
    chain_scenarios_data = []
    num=1
    
    for scenario in scenario_list:
        assert (scenario.is_file() and scenario.suffix
                == ".json"), "scenario file must be a json file"
        with open(scenario, "r", encoding="utf-8") as f:  # *********
            scenario_data: dict = json.loads(f.read())[0]
        scenario_data["ip"]=f"192.168.1.{num}"
        num+=1
        chain_scenarios_data.append(scenario_data)

    if chain_length > len(chain_scenarios_data):
        console.print("[red]chain_length is too large")
        return
    elif chain_length <= 0:
        chain_length = len(chain_scenarios_data)
        sampled_chain_scenarios_data = random.shuffle(chain_scenarios_data)

        file_path = (
            saved_path /
            f"chain-{str(source_scenario_path.stem)}-full-{chain_length}_envs-seed_{seed}.json"
        )
    else:
        sampled_chain_scenarios_data = random.sample(chain_scenarios_data,
                                                      k=chain_length)
        file_path = (
            saved_path /
            f"chain-{str(source_scenario_path.stem)}-sample-{chain_length}_envs-seed_{seed}.json"
        )
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(sampled_chain_scenarios_data, f, indent=4)
        console.print(f"[green]save chain scenarios to {file_path}")
    return sampled_chain_scenarios_data, file_path


if __name__ == "__main__":
    chain_scenarios_path = scenario_path / "chain"

    chain_scenarios_path.mkdir(parents=True, exist_ok=True)

    create_chain_scenarios(source_scenario_path=scenario_path / "msfexp_vul",
                           chain_length=5)
