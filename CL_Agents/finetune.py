import sys
import os
from loguru import logger as logging

from torch.utils.tensorboard import SummaryWriter
from agent import Agent
from RL_Policy.config import config, Finetune_Config


class Player(Agent):

    def __init__(self,
                 logger,
                 use_wandb=False,
                 policy_name="PPO",
                 seed=0,
                 config: config = None):
        super().__init__(logger=logger,
                         use_wandb=use_wandb,
                         policy_name=policy_name,
                         seed=seed,
                         config=config)

    def reset(self):
        self.Policy.actor.reset()
        self.Policy.critic.reset()
        # self.Policy.memory.count = 0
        logging.info("Reset the model...")
class FinetuneAgent():

    def __init__(self,
                 logger: SummaryWriter,
                 use_wandb=False,
                 policy_name="PPO",
                 seed=0,
                 config=None,
                 cl_config=None):
        # ft agent的config是使用policy config初始化的
        self.use_wandb = use_wandb
        self.player = Player(logger=None,
                             use_wandb=self.use_wandb,
                             policy_name=policy_name,
                             seed=seed,
                             config=cl_config)
        
        self.config = self.player.config
        if not cl_config:
            self.cl_config = Finetune_Config()
        else:
            self.cl_config = cl_config

        self.name = self.cl_config.name
        self.logger = logger

    def get_new_task_learner(self, new_task_id):
        self.task_id = new_task_id
        # self.player.Policy.memory.count = 0
        if new_task_id != 0 and self.cl_config.reset:
            self.player.reset()

        return self.player

    def save(self, path):
        self.player.save(path)

    def policy_preservation(self, all_task,verbose=False):

        return

    def get_task_evaluator(
        self,
        on_train=False,
    ):
        return self.player

    