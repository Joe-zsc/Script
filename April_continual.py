import os
from loguru import logger as logging
import numpy as np
import wandb
import sys

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
sys.path.append(curr_path)  # add current terminal path to sys.path
isDebug = True if sys.gettrace() else False

# tensorboard --logdir
# tensorboard --logdir runs --host localhost --port 6006

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--env_file",
                        default=None,
                        help="environment file, e.g. train.json")
    parser.add_argument("--cl_train_num",
                        type=int,
                        default=6 ,
                        help="task number for continual learning")
    parser.add_argument("--cl_method",
                        default="ft",
                        choices=["", "ft", "script"])

    parser.add_argument("--config_file", default=None, type=str)

    parser.add_argument(
        "--use_wandb",
        action='store_true',
        default=False,
    )
    parser.add_argument("--use_tensorboard",
                        action='store_true',
                        default=True,
                        help="save tensorboard log")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--note", type=str, default='', help="wandb note")
    parser.add_argument("--gpu", type=str, default='0')

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    from util import UTIL, Configure, set_seed
    from Bot import BOT
    UTIL.show_banner()
    
    logging.add(
        UTIL.log_path / f"{UTIL.project_name}.log",
        encoding='UTF-8',
        colorize=True,
        level='INFO',
        backtrace=True,
        diagnose=True,
        format=
        '{time:YYYY-MM-DD HH:mm:ss} - {level} - {file} - {line} - {message}')
    
    # ---------------------------------------------------------------------------- #
    #                                  Environment
    # ---------------------------------------------------------------------------- #
    scenario_path = UTIL.project_path / "scenarios"
    if not args.env_file:
        assert args.cl_train_num, "[cl_train_num] must be set if [env_file] is not set"
        from scenarios.create_chain_scenarios import create_chain_scenarios
        sampled_chain_scenarios_data, file_path = create_chain_scenarios(
            source_scenario_path=scenario_path / "msfexp_vul",
            seed=args.seed,
            chain_length=int(args.cl_train_num)) # the last env is used for eval forward transfer performance
        
        args.env_file = file_path
    else:
        if not args.env_file.endswith(".json"):
            args.env_file = args.env_file + ".json"
        args.env_file = scenario_path / args.env_file

    # ---------------------------------------------------------------------------- #
    #                                     Start                                    #
    # ---------------------------------------------------------------------------- #

    set_seed(args.seed)
    Bot = BOT(**vars(args))

    if Bot.use_wandb:
        try:
            os.environ["WANDB_BASE_URL"] = Configure.get(
                'wandb', 'WANDB_BASE_UR')
            wandb.login(key=Configure.get('wandb', 'API_Key'))
        except:
            os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
        run_mode = "debug" if isDebug else "common"
        Bot.wandb_run = wandb.init(project=UTIL.project_name,
                                   name=Bot.title,
                                   notes=args.note,
                                   tags=["test"],
                                   job_type=f"{run_mode}",
                                   group=f"{Bot.agent.cl_agent.name}",
                                   reinit=True,
                                   allow_val_change=True,
                                   config=Bot.running_config,
                                   save_code=False)

    Bot.train()

    logging.info("😉  Have a good day~")
    wandb.finish()
