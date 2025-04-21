import os
import platform
import copy
import torch
import json
from pprint import pprint, pformat
import time
from loguru import logger as logging
import pandas as pd
from prettytable import PrettyTable
from datetime import datetime
from util import Configure, UTIL, color, Matrix, console
# from agent import Agent
from agent_continual import Agent_CL
from agent import Agent
from actions import *
from host import HOST, StateEncoder
import wandb
from pathlib import Path
from rich.pretty import Pretty, pprint
from rich.panel import Panel
import asciichartpy
from NLP_Module.Encoder import *
# tensorboard --logdir runs --host localhost --port 8896
from torch.utils.tensorboard import SummaryWriter


class BOT:
    """ Deep Q-Network Bot """

    def __init__(self,
                 mode=0,
                 env_file=None,
                 cl_method="",
                 cl_train_num=40,
                 policy: str = "PPO",
                 config_file='',
                 config=None,
                 cl_config=None,
                 use_wandb=False,
                 use_tensorboard=True,
                 save=False,
                 seed=0,
                 note='',
                 testing_args=None,
                 **kwargs):
        self.use_wandb = use_wandb
        self.wandb_run = None
        self.host_name = f"{platform.platform()}-{platform.node()}"
        self.env_file = Path(env_file)
        self.policy_name = policy
        self.cl_method = cl_method
        self.time_flag = datetime.now().strftime('%b%d_%H-%M-%S')
        self.cl_train_num = cl_train_num
        self.testing_args = testing_args
        if use_tensorboard:
            self.tensorboard_logger = SummaryWriter()
        else:
            self.tensorboard_logger = None
        self.config_file = config_file

        if cl_method:
            self.agent = Agent_CL(method=self.cl_method,
                                  use_wandb=self.use_wandb,
                                  time_flag=self.time_flag,
                                  logger=self.tensorboard_logger,
                                  policy_name=self.policy_name,
                                  config=config,
                                  cl_config=cl_config,
                                  config_file=self.config_file,
                                  seed=seed)
            self.title = f"{self.agent.name}-{self.time_flag}-{self.env_file.stem}-{self.cl_train_num}"
        else:
            self.agent = Agent(policy_name=self.policy_name,
                               use_wandb=self.use_wandb,
                               logger=self.tensorboard_logger,
                               config=config,
                               seed=seed)
            self.title = f"{self.agent.name}-{self.time_flag}-{self.env_file.stem}-{seed}"
        # if testing_args:
        #     logging.info(f"Testing args : {testing_args}")
        self.load_agent = ''
        self.save_model = save

        self.note = note
        self.seed = seed
        self.running_config = self.get_running_config()
        # UTIL.line_break(length=80, symbol='=')
        logging.info(f"Bot Created: {self.title}")
        # UTIL.line_break(length=80, symbol='=')

    def __del__(self):

        print(f"Bot {self.time_flag} Deleted")

    def get_running_config(self):
        config_to_wandb = copy.deepcopy(self.agent.config.__dict__)
        if self.cl_method:
            config_to_wandb.update(self.agent.cl_config.__dict__)
        config_to_wandb["Algo"] = self.policy_name
        config_to_wandb["action_set"] = Action.vul_hub_path.name
        config_to_wandb["env_name"] = self.env_file.stem
        config_to_wandb["load_agent"] = self.load_agent
        config_to_wandb["cl_method"] = self.cl_method
        config_to_wandb["seed"] = self.seed
        config_to_wandb["state_dim"] = StateEncoder.state_space
        config_to_wandb["action_dim"] = Action.action_space
        config_to_wandb["config_file"] = self.config_file
        if self.load_agent:
            config_to_wandb["loaded"] = True
        else:
            config_to_wandb["loaded"] = False
        config_df = pd.DataFrame.from_dict(config_to_wandb, orient='index')
        return config_to_wandb

    def make_env(self, env_file=None):
        target_list: list[HOST] = []
        env_vuls = []
        with open(env_file, 'r', encoding='utf-8') as f:  # *********
            self.environment_data = json.loads(f.read())
            train_ip_list = []
            for host in self.environment_data:
                ip = host["ip"]
                assert ip not in train_ip_list, f"{ip} aready exist in {env_file}"
                train_ip_list.append(ip)
                vul = host["vulnerability"][0]
                if vul not in Action.Vul_cve_set:
                    logging.error(f"host vul {vul} is not exploitable")
                    exit(0)
                t = HOST(ip, env_data=host, env_file=env_file)
                env_vuls.append(vul)
                target_list.append(t)

        return target_list

    def train(self):
        eval_FWT = self.cl_method in ["ft", "script"]

        logging.info("Starting training")
        env = self.make_env(self.env_file)

        # random.shuffle(train_env)
        if eval_FWT:
            train_env = env[:self.cl_train_num - 1]
            fwt_eval_task = env[
                -1]  # the last env is used for eval forward transfer performance
            assert fwt_eval_task not in train_env
            if self.use_wandb:
                wandb.config.update({"FWT_eval_env": fwt_eval_task.env_data})
        else:
            train_env = env[:self.cl_train_num]
            fwt_eval_task = None
        # save training env data
        self.train_env_data = []
        for e in train_env:
            self.train_env_data.append(e.env_data)

        logging.debug(self.running_config)
        console.print(
            Panel(Pretty(self.running_config),
                  expand=False,
                  title="Train parameters"))
        start = time.time()

        UTIL.Running_title = self.title
        if self.cl_method:

            train_matrix = self.agent.train_continually(
                task_list=train_env, forward_transfer_eval_task=fwt_eval_task)
            if train_matrix:
                logging.info("Learning Curve of SR_previous_tasks")
                self.plot_reward(data=train_matrix.SR_previous_tasks)
        else:
            train_matrix = self.agent.train_with_tqdm(task_list=env)
            if train_matrix:
                logging.info("Learning Curve of Train_Episode_Rewards")
                self.plot_reward(data=train_matrix.Train_Episode_Rewards)
        end = time.time()

        if self.wandb_run:
            self.wandb_run.tags += (train_matrix.signal, )

        # eval_sr = self.Eval_Simulate(verbose=verbose)
        cfg = self.log_paras(time=self.time_flag)
        # if self.save_model:
        #     if eval_sr > 0.99:
        if train_matrix.signal == Matrix.Finished:
            self.save_experiment_record(cfg=cfg)
            logging.success(f"{self.time_flag} training complete.")
        else:
            logging.warning(f"{self.time_flag} {train_matrix.signal}.")

        run_time = time.strftime("%H:%M:%S", time.gmtime(round(end - start)))
        logging.info(f"Running Time: {run_time}")
        self.train_matrix = train_matrix

    def Eval_Simulate(self, eval_times=1, interactive=False, verbose=True):

        mean_eval_rewads = 0
        mean_success_rate = 0.0
        i = 0
        env = self.make_env(env_file=self.env_file)

        while i < eval_times:

            attack_path, eval_rewards, eval_sr = self.agent.Evaluate(
                target_list=env,
                interactive=interactive,
                verbose=verbose,
                step_limit=10)

            for host_attack_path in attack_path:
                table = PrettyTable(host_attack_path[0].keys())
                table.title = f"{host_attack_path[0]['target']}"
                for process in host_attack_path:
                    table.add_row([
                        process["target"], process["step"], process["action"],
                        process["result"], process["reward"]
                    ])
                if verbose:
                    print(table)
            mean_eval_rewads += eval_rewards
            mean_success_rate += eval_sr
            i += 1
            print(f"Evaluation times : #{i}")
            print(
                f"evaluation rewards = {color.color_str(eval_rewards,c=color.GREEN)}"
            )
            print(f"success_rate = {color.color_str(eval_sr,c=color.GREEN)}")
        mean_eval_rewads = mean_eval_rewads / eval_times
        mean_success_rate = mean_success_rate / eval_times
        if eval_times > 1:
            print(
                f"Mean evaluation rewards = {color.color_str(mean_eval_rewads,c=color.GREEN)}"
            )
            print(
                f"Mean success_rate = {color.color_str(mean_success_rate,c=color.GREEN)}"
            )
        return mean_success_rate

    def log_paras(self, time, log_file=None):
        legal_actions = Action.legal_actions
        logs = []

        algo_cfg = self.agent.config.__dict__.copy()
        algo_cl_cfg = self.agent.cl_config.__dict__.copy(
        ) if self.cl_method else {}
        cfg = {}
        cfg["time"] = time
        cfg["policy"] = self.agent.policy_name
        cfg["cl_method"] = self.cl_method
        cfg["note"] = self.note
        cfg["seed"] = self.seed
        cfg["load_agent"] = self.load_agent
        cfg["eval_reward"] = self.agent.eval_rewards
        cfg["eval_success_rate"] = self.agent.eval_success_rate

        cfg["env"] = self.env_file.parent.name + '/' + self.env_file.stem
        cfg["host_name"] = self.host_name
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        cfg.update(algo_cfg)
        cfg.update(algo_cl_cfg)

        cfg["action_space"] = Action.action_space
        cfg["state_space"] = StateEncoder.state_space
        cfg["state_vector"] = StateEncoder.state_vector
        cfg["SBERT_model_name"] = Configure.get("Embedding", "sbert_model")
        cfg["vul_hub"] = Action.vul_hub_path.stem

        return cfg

    def save_experiment_record(self, cfg):
        path = UTIL.log_path / "experiment_record" / self.title
        if not os.path.exists(path):
            os.makedirs(path)
        #1 scenario
        UTIL.save_json(path=path / f"scenario.json", data=self.train_env_data)

        #2 parameters
        UTIL.save_json(path=path / f"parameters.json", data=cfg)

        #3 agent model
        # self.agent.save(path)
        logging.success(f"experiment record saved in path : {path}")

    def plot_reward(self,
                    data: list,
                    smooth=True,
                    width=100,
                    smooth_weight=0.8):
        '''
        width: number of sampled points
        
        '''
        rewards = UTIL.smooth_data(data,
                                   weight=smooth_weight) if smooth else data
        length = len(rewards)
        iter = length // width if length > width else 1
        logging.info(f"\n{asciichartpy.plot(rewards[0:length:iter], {
                'height': 10,
                'max': max(data) * 1.1,
                'min': min(data) -0.1*min(data)
            })}"
            )
