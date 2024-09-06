import os
from loguru import logger as logging
from pathlib import Path
import torch
import random
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



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--cl_train_num",
                        type=int,
                        default=40,
                        help="training data set, e.g. train.json")
    parser.add_argument("--mode",
                        default="t",
                        help="training data set, e.g. train.json")
    parser.add_argument("--cl_method",
                        default="script",
                        choices=["", "ft", "pc", "p&c", "script", "medqn"],
                        help="support PPO D3QN")
    parser.add_argument("--policy", default="PPO", help="support PPO D3QN")
    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        help="support PPO D3QN")
    parser.add_argument("--load",
                        default="",
                        type=str,
                        help="support PPO D3QN")
    parser.add_argument("--save",
                        action='store_true',
                        default=False,
                        help="support PPO D3QN")
    parser.add_argument("--use_wandb",
                        action='store_true',
                        default=True,
                        help="support PPO D3QN")
    parser.add_argument("--use_tensorboard",
                        action='store_true',
                        default=False,
                        help="support PPO D3QN")
    parser.add_argument("--seed", type=int, default=0, help="support PPO D3QN")
    parser.add_argument("--note",
                        type=str,
                        default='',
                        help="support PPO D3QN")
    parser.add_argument("--gpu",
                        type=str,
                        default='0',
                        help="support PPO D3QN")
    parser.add_argument(
        "--paras",
        # nargs='+',
        # type=dict,
        required=False,
        default=None,
        help="support PPO D3QN")
    args = parser.parse_args()
    seed = args.seed
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  
    args.use_wandb = not isDebug
    # set seed

    from util import UTIL, Configure
    from Bot import BOT

    logging.add(
        UTIL.log_path / f"{UTIL.project_name}.log",
        encoding='UTF-8',
        colorize=True,
        level='INFO',
        backtrace=True,
        diagnose=True,
        format=
        '{time:YYYY-MM-DD HH:mm:ss} - {level} - {file} - {line} - {message}')

    UTIL.show_banner()

    # ---------------------------------------------------------------------------- #
    #                             create training tasks                            #
    # ---------------------------------------------------------------------------- #

    total_host_scenario = Path("scenarios/all_scenario_msf.json")
    scenario_path = UTIL.project_path / "scenarios/CL_scenarios"
    env_file = f"{total_host_scenario.stem}-{args.cl_train_num+1}-seed_{seed}.json"
    env_file = scenario_path / env_file
    #   TODO: 顺序是否要随机化后续再定
    if not os.path.exists(env_file):
        from generate_scenario import generate_chain_scenario
        targets = generate_chain_scenario(chain_length=args.cl_train_num + 1,
                                          all_target_file=total_host_scenario,
                                          saving_scenario_file=env_file,
                                          mode="random")

    set_seed(seed)
    Bot = BOT(**vars(args), env_file=env_file)

    if args.use_wandb:
        try:
            os.environ["WANDB_BASE_URL"] = Configure.get(
                'wandb', 'WANDB_BASE_UR')
            wandb.login(key=Configure.get('wandb', 'API_Key'))
        except:
            os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
        run_mode = "debug" if isDebug else "common"
        Bot.wandb_run = wandb.init(
            project=UTIL.project_name,
            name=Bot.title,
            notes=args.note,
            tags=["exp","fwt"],
            job_type=f"{run_mode}",
            group=
            f"{Bot.agent.cl_agent.name}",  #RQ-1-April-1000-K10-UCB-test-03311503
            reinit=True,
            allow_val_change=True,
            config=Bot.running_config,
            save_code=False)

    if args.load:
        Bot.load(agent_name=args.load)

    if args.mode == "t":

        Bot.train(verbose=False)

    elif args.mode == "e":

        Bot.Eval_Simulate(verbose=True, interactive=True)

    logging.info("😉  Have a good day~")
    wandb.finish()
