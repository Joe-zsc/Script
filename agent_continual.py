import sys
import os
from loguru import logger as logging
import wandb
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from easydict import EasyDict
import copy

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
from util import UTIL, color, Matrix, Merge_str_lst
from RL_Policy.config import *
from agent import Agent
import CL_Agents


class Agent_CL():
    def __init__(self,
                 time_flag,
                 logger: SummaryWriter = None,
                 use_wandb=False,
                 method="finetune",
                 policy_name="PPO",
                 seed=0,
                 config: config = None,
                 cl_config=None,
                 config_file=None):
        self.tf_logger = logger
        self.use_wandb = use_wandb
        self.method = method
        self.policy_name = policy_name
        self.seed = seed
        self.time_flag = time_flag
        if config_file:
            cl_config = self.read_config_file(config_file)
        self.cl_agent = self._get_cl_agent(config=config, cl_config=cl_config)
        self.name = self.cl_agent.name
        self.config = self.cl_agent.config
        self.cl_config = self.cl_agent.cl_config
        self.eval_rewards = 0
        self.eval_success_rate = 0
        self.current_task_id = -1
        self.FWT_eval_tasks_num = 0
        self.wandb_run = None

    def read_config_file(self, config_file):
        #! note: config_file is conly used for cl_config currently
        config_file = UTIL.project_path / "config" / config_file
        cl_config = UTIL.read_yaml(config_file)
        if self.method != cl_config["name"]:
            self.method = cl_config["name"]
            logging.warning(
                f"continual learning method is changed to {cl_config['name']}")
        cl_config = self.get_config(self.method, cl_config["parameters"])
        return cl_config

    def get_config(self, method_name, args: dict):
        if method_name in ["EWC", "ewc"]:
            return EWC_Config(**args)

        elif method_name in ["ft", "finetune"]:
            return Finetune_Config(**args)

        elif method_name in ["p&c"]:
            return ProgressCompress_Config(**args)

        elif method_name in ["pc"]:
            return PolicyConsolidation_Config(**args)

        elif method_name in ["script"]:
            return Script_Config(**args)

        elif method_name in ["medqn"]:
            return MeDQN_Config(**args)

    def _get_cl_agent(self, config, cl_config):

        

        if self.method in ["script"]:
            return CL_Agents.ScriptAgent(logger=self.tf_logger,
                                         use_wandb=self.use_wandb,
                                         policy_name=self.policy_name,
                                         seed=self.seed,
                                         config=config,
                                         cl_config=cl_config)



    def train_continually(self,
                          task_list,
                          forward_transfer_eval_task=None,
                          forward_transfer_eval_freq=5,
                          eval_freq=5,
                          eval_all_task=False,
                          save_agent=False,
                          verbose=False):

        CL_Train_matrix = EasyDict({
            "signal": Matrix.Finished,
            "Rewards_initial_task": [],
            "SR_previous_tasks": [],
            "Rewards_current_task": [],
            "last_task": -1
        })
        eval_first_task_rewards, eval_current_task_rewards, all_tasks_sr = 0, 0, 0

        logging.info(f"Train continually using {self.cl_agent.name}")

        save_path = UTIL.trained_agent_path / self.name / self.policy_name / self.time_flag
        # for i in tqdm(range(len(target_list)), desc="Total", position=0):
        for i in range(len(task_list)):
            self.current_task_id = i
            '''
            # ---------------------------------------------------------------------------- #
            #                               learn a new task                               #
            # ---------------------------------------------------------------------------- #
            '''
            new_task_learner = self.cl_agent.get_new_task_learner(
                new_task_id=i)
            _new_task_learner_ = copy.deepcopy(new_task_learner)
            start = time.time()
            result, Task_Train_matrix = self.learn_new_task(
                player=new_task_learner,
                all_task=task_list,
                eval_freq=eval_freq,
                eval_all_task=eval_all_task)
            end = time.time()
            run_time = float(end - start)
            # early terminated if the agent failed to learn a new task
            if Task_Train_matrix.Train_Episode_Rewards[
                    -1] < 900 and not result["success"]:
                # if not result["success"]:
                logging.error(f"learning task {i} failed")
                logging.error(f"env:{task_list[i].env_data['vulnerability']}")
                # CL_Train_matrix.signal = Matrix.Failed
                # break
            '''
            # ---------------------------------------------------------------------------- #
            #                         performe policy preservation                        #
            # ---------------------------------------------------------------------------- #
            '''
            start = time.time()
            self.cl_agent.policy_preservation(all_task=task_list,
                                              verbose=verbose)
            end = time.time()
            policy_preservation_run_time = float(end - start)
            #-------save the agent--------

            # color.print(s=f"saving agent, path = {save_path}",c=color.RED)
            if save_agent:
                self.cl_agent.save(path=save_path / f"task_{i}")
            '''
            # ---------------------------------------------------------------------------- #
            #                  evaluate forgetting performance after training a task                  #
            # ---------------------------------------------------------------------------- #
            '''
            evaluate_player = self.cl_agent.get_task_evaluator(on_train=False)
            attack_path, eval_first_task_rewards, eval_current_task_rewards, all_tasks_sr = self.Evaluate(
                target_list=task_list[:i + 1],
                player=evaluate_player,
                verbose=False,
                step_limit=self.config.eval_step_limit)
            failed_list = []
            for t in range(len(attack_path)):
                if not attack_path[t]["success"]:
                    failed_list.append(t)
            logging.info(
                f"After learning task {color.color_str(i,c=color.RED)}, Previous Tasks SR: {color.color_str(all_tasks_sr,c=color.BLUE)}, failed_list: {Merge_str_lst(failed_list)},task_total_steps:{Task_Train_matrix.task_total_steps}"
            )
            if self.tf_logger:
                # self.tf_logger.add_scalar(
                #     "Eval_after_Training/Rewards of initial task",
                #     eval_first_task_rewards, i)
                # self.tf_logger.add_scalar(
                #     "Eval_after_Training/Rewards of current task",
                #     eval_current_task_rewards, i)
                self.tf_logger.add_scalar(
                    "Eval_after_Training/SR of previous tasks", all_tasks_sr,
                    i)
                self.tf_logger.add_scalar("Time/CL_PerTask", run_time, i)
            if self.use_wandb:
                wandb.log({
                    # "Eval_after_Training/Rewards_of_Initial_Task":
                    # eval_first_task_rewards,
                    # "Eval_after_Training/Rewards_of_Current_Task":
                    # eval_current_task_rewards,
                    "Eval_after_Training/SR_of_Previous_Task":
                    all_tasks_sr,
                    "Eval_after_Training/task_total_steps":
                    Task_Train_matrix.task_total_steps,
                    # "Time/CL_PerTask":
                    # run_time,
                    # "Time/policy_preservation_PerTask":
                    # policy_preservation_run_time,
                    # "Eval_after_Training/First_episode_reward":
                    # Task_Train_matrix.Train_Episode_Rewards[0],
                    "task":
                    i
                })

            #--------------------------------------------------------------------------------
            CL_Train_matrix.Rewards_initial_task.append(
                eval_first_task_rewards)
            CL_Train_matrix.SR_previous_tasks.append(all_tasks_sr)
            CL_Train_matrix.Rewards_current_task.append(
                eval_current_task_rewards)
            CL_Train_matrix.last_task = i
            ####    Early Quit setting    ####
            if Matrix.Train_Matrix.best:
                if i < len(Matrix.Train_Matrix.best.SR_previous_tasks):
                    if all_tasks_sr < Matrix.Train_Matrix.best.SR_previous_tasks[
                            i]:

                        logging.warning(
                            f"SR_of_Previous_Task {all_tasks_sr} < {Matrix.Train_Matrix.best.SR_previous_tasks[i]}"
                        )
                        # CL_Train_matrix.signal = Matrix.EarlyTerminate
                        # break
            # if i in [10, 15, 20, 25, 30,35]:
            # if i >10:
            #     if all_tasks_sr < 0.5 and self.method in ["script"]:
            #         CL_Train_matrix.signal = Matrix.EarlyTerminate
            #         logging.warning(f"learning task {i} terminated")
            #         break

            ########################################
            '''
            # ---------------------------------------------------------------------------- #
            #                     evaluate forward transfer performance                    #
            # ---------------------------------------------------------------------------- #
            '''
            if i in [1, 5, 10, 20, 30] and forward_transfer_eval_task:

                task_total_steps = self.Forward_transfer_evaluate(
                    player=_new_task_learner_,
                    task=[forward_transfer_eval_task])

        if forward_transfer_eval_task and i == len(task_list) - 1:
            i += 1
            self.current_task_id = i
            _new_task_learner_ = self.cl_agent.get_new_task_learner(
                new_task_id=i)
            task_total_steps = self.Forward_transfer_evaluate(
                player=_new_task_learner_, task=[forward_transfer_eval_task])
        self.eval_rewards = eval_first_task_rewards
        self.eval_success_rate = all_tasks_sr

        Matrix.Train_Matrix.all.append(CL_Train_matrix)
        if CL_Train_matrix.signal == Matrix.Finished:
            Matrix.Train_Matrix.best = CL_Train_matrix
        return CL_Train_matrix

    def reset_player_for_newtask(self, player: Agent):
        player.task_num_episodes = 0
        player.total_training_step = 0
        player.best_return = -float('inf')
        player.best_action_set = []
        player.best_episode = 0
        player.best_reward_episode = []
        player.eval_rewards = 0
        player.eval_success_rate = 0

    def learn_new_task(self,
                       player: Agent,
                       all_task,
                       eval_freq,
                       eval_all_task=False):

        eval_first_task_rewards, eval_current_task_rewards, previous_tasks_sr = 0, 0, 0
        task_total_steps = 0
        Task_Train_matrix = EasyDict({
            "Train_Episode_Rewards": [],
            "Train_Episode_Steps": [],
            "Train_Success_Rate": [],
            "Train_Episode_Time": [],
            "Eval_First_task_Rewards": [],
            "Eval_Current_task_Rewards": [],
            "Eval_Previous_tasks_SR": [],
            "task_total_steps": 0,
            "last_task": -1
        })
        # Train_Episode_Rewards = []
        # Train_Episode_Steps = []
        # Train_Success_Rate = []
        # Train_Episode_Time = []
        # Eval_First_task_Rewards = []
        # Eval_Previous_tasks_SR = []
        task = [all_task[self.current_task_id]]
        self.reset_player_for_newtask(player=player)
        with tqdm(
                range(self.config.train_eps),
                colour='green',
                desc=
                f"{color.color_str(f'{self.method}-Training Task {self.current_task_id}',c=color.RED)}"
        ) as tbar:
            for _ in tbar:
                start = time.time()
                player.num_episodes += 1

                ep_results = player.run_train_episode(
                    task, update_norm=(self.current_task_id == 0))

                # if player.num_episodes <= 10:
                #     self.logger.add_text(
                #         "episode_actions", ' , '.join(
                #             list(map(lambda x: str(x), player.action_set))),
                #         player.num_episodes)
                end = time.time()
                run_time = float(end - start)
                ep_return, ep_steps, success_rate = ep_results
                Task_Train_matrix.task_total_steps += ep_steps
                player.last_episode_reward = ep_return
                Task_Train_matrix.Train_Episode_Rewards.append(ep_return)
                # Task_Train_matrix.Train_Episode_Steps.append(ep_steps)
                # Task_Train_matrix.Train_Success_Rate.append(success_rate)
                # Task_Train_matrix.Train_Episode_Time.append(run_time)
                if self.tf_logger:
                    self.tf_logger.add_scalar("Train/Episode Rewards",
                                              ep_return, player.num_episodes)
                    self.tf_logger.add_scalar("Train/Episode Steps", ep_steps,
                                              player.num_episodes)

                    self.tf_logger.add_scalar("Time/Episode_Time", run_time,
                                              player.num_episodes)
                    if player.use_state_norm:
                        self.tf_logger.add_scalar(
                            "Auxillary/state_norm_mean",
                            player.state_norm.running_ms.mean.mean(),
                            player.num_episodes)
                    if player.use_lr_decay:
                        self.tf_logger.add_scalar(
                            "Auxillary/lr_actor",
                            player.Policy.actor_optimizer.state_dict()
                            ['param_groups'][0]['lr'], player.num_episodes)
                        self.tf_logger.add_scalar(
                            "Auxillary/lr_critic",
                            player.Policy.critic_optimizer.state_dict()
                            ['param_groups'][0]['lr'], player.num_episodes)
                if self.use_wandb:
                    wandb.log({
                        "Total_Episode_per_Task":
                        player.num_episodes / player.config.train_eps,
                        "Total_Episode":
                        player.num_episodes,
                        "Train/Train_Episode_Rewards":
                        ep_return,
                        "Train/Train_Episode_Steps":
                        ep_steps,
                        "Time/Episode_Time":
                        run_time,
                        #########
                        # "memory_size":
                        # player.Policy.replaymemory.size,
                    })

                if player.num_episodes % eval_freq == 0:
                    player.eval_times += 1
                    eval_tasks = task if not eval_all_task else all_task[:self.
                                                                         current_task_id
                                                                         + 1]
                    start = time.time()
                    _, eval_first_task_rewards, eval_current_task_rewards, previous_tasks_sr = self.Evaluate(
                        target_list=eval_tasks,
                        # target_list=task,
                        player=self.cl_agent.get_task_evaluator(on_train=True),
                        verbose=False,
                        step_limit=player.config.eval_step_limit)
                    end = time.time()
                    Eval_on_Training_run_time = float(end - start)

                '''
                display info
                '''
                if not eval_all_task:
                    previous_tasks_sr = "/"
                    eval_first_task_rewards = '/'
                tbar.set_postfix(
                    r=color.color_str(f"{ep_return}/{player.best_return}",
                                      c=color.PURPLE),
                    step=color.color_str(f"{ep_steps}", c=color.DARKCYAN),
                    e_i_r=color.color_str(f"{eval_first_task_rewards}",
                                          c=color.BLUE),
                    e_p_sr=color.color_str(f"{previous_tasks_sr}",
                                           c=color.CYAN),
                    e_c_r=color.color_str(f"{eval_current_task_rewards}",
                                          c=color.GREEN),
                )
        attack_path, _, _, _ = self.Evaluate(
            target_list=all_task[:self.current_task_id + 1],
            player=self.cl_agent.get_task_evaluator(on_train=True),
            verbose=False,
            step_limit=player.config.eval_step_limit)
        return attack_path[-1], Task_Train_matrix

    def Forward_transfer_evaluate(self, player: Agent, task, eval_freq=5):
        self.FWT_eval_tasks_num += 1
        FWT_num_episodes = 0
        eval_first_task_rewards, eval_current_task_rewards, previous_tasks_sr = 0, 0, 0
        task_total_steps = 0
        self.reset_player_for_newtask(player=player)
        player.use_wandb = False
        with tqdm(
                range(self.config.train_eps),
                colour='yellow',
                desc=
                f"{color.color_str(f'{self.method}-FWT-{self.current_task_id}',c=color.YELLOW)}"
        ) as tbar:
            for _ in tbar:
                start = time.time()
                FWT_num_episodes += 1

                ep_results = player.run_train_episode(
                    task, update_norm=(self.current_task_id == 0))

                end = time.time()
                run_time = float(end - start)
                ep_return, ep_steps, success_rate = ep_results
                task_total_steps += ep_steps
                if self.tf_logger:
                    self.tf_logger.add_scalar(
                        f"FWT/Episode_Rewards_after_task_{self.current_task_id}",
                        ep_return, FWT_num_episodes)
                    # self.tf_logger.add_scalar(
                    #     f"FWT/Episode_Steps_after_task_{self.current_task_id}",
                    #     ep_steps, evaluate_num_episodes)

                    # self.tf_logger.add_scalar(
                    #     f"FWT/Episode_Time_after_task_{self.current_task_id}",
                    #     run_time, evaluate_num_episodes)

                if self.use_wandb:
                    wandb.log({
                        "FWT_num_episodes":
                        FWT_num_episodes,
                        f"FWT/Train_Episode_Rewards_after_task_{self.current_task_id}":
                        ep_return,
                        # f"FWT/Train_Episode_Steps_after_task_{self.current_task_id}":
                        # ep_steps,
                        # f"FWT/Episode_Time_after_task_{self.current_task_id}":
                        # run_time,
                    })

                if FWT_num_episodes % eval_freq == 0:
                    _, eval_first_task_rewards, eval_current_task_rewards, previous_tasks_sr = self.Evaluate(
                        target_list=task,
                        player=player,
                        verbose=False,
                        step_limit=player.config.eval_step_limit)
                '''
                display info
                '''

                tbar.set_postfix(
                    r=color.color_str(f"{ep_return}/{player.best_return}",
                                      c=color.PURPLE),
                    step=color.color_str(f"{ep_steps}", c=color.DARKCYAN),
                    e_c_r=color.color_str(f"{eval_current_task_rewards}",
                                          c=color.GREEN),
                )
            if self.use_wandb:
                wandb.log({
                    "FWT/task_total_steps": task_total_steps,
                    # "Time/CL_PerTask":
                    # run_time,
                    # "Time/policy_preservation_PerTask":
                    # policy_preservation_run_time,
                    # "Eval_after_Training/First_episode_reward":
                    # Task_Train_matrix.Train_Episode_Rewards[0],
                    "FWT_eval_tasks_num": self.FWT_eval_tasks_num
                })

            return task_total_steps

    def Evaluate(self,
                 target_list,
                 player: Agent,
                 interactive=False,
                 step_limit=10,
                 verbose=True):
        attack_path, total_rewards, total_tasks_sr = player.Evaluate(
            target_list=target_list,
            step_limit=step_limit,
            verbose=verbose,
            interactive=interactive)
        eval_first_task_rewards = attack_path[0]["reward"]
        eval_current_task_rewards = attack_path[-1]["reward"]
        return attack_path, eval_first_task_rewards, eval_current_task_rewards, total_tasks_sr
