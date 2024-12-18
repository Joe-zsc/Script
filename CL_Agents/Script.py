import sys
import os
from loguru import logger as logging
import time
from tqdm import tqdm
import wandb
import copy
import torch
from torch.optim import Adam, SGD

import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions.kl import kl_divergence
import numpy as np
import random
from typing import Dict
import itertools
from torch.utils.tensorboard import SummaryWriter

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
sys.path.append(curr_path)  # add current terminal path to sys.path
import torch.nn as nn
from util import UTIL, color
from agent import Agent
from RL_Policy.config import config, Script_Config
from RL_Policy.common import Memory
from host import HOST
from collections import namedtuple, defaultdict
from cl_method import ParamData, zerolike_params_dict, copy_params_dict
from RL_Policy.PPO import Actor, Critic, PPO_agent
from RL_Policy.config import *




class ExplorePolicy(PPO_agent):
    def __init__(self, cfg: PPO_Config, logger: SummaryWriter, use_wandb):
        super().__init__(cfg, logger, use_wandb)
        self.AEI = 0
        self.last_delta = 0

    def update_policy(self,
                      num_episode,
                      train_steps,
                      guide_policy=None,
                      temperature=1,
                      guide_kl_scale=1,
                      use_grad_clip=True):
        if self.memory.count == self.batch_size:
            _, _, delta = self.update(train_steps, guide_policy,
                                      guide_kl_scale, temperature,
                                      use_grad_clip)
            # self.AEI = delta * delta - self.last_delta * self.last_delta
            self.last_delta = delta
            self.memory.count = 0

    def calcuate_ppo_loss(self,
                          s_minibatch,
                          a_minibatch,
                          adv_minibatch,
                          a_logprob_minibatch,
                          v_target_minibatch,
                          guide_policy=None,
                          guide_kl_scale=1,
                          temperature=1):
        logits = self.actor.net(s_minibatch)
        probs = F.softmax(logits, dim=-1)
        dist_now = Categorical(probs=probs)
        dist_entropy = dist_now.entropy().view(-1,
                                               1)  # shape(mini_batch_size X 1)
        a_logprob_now = dist_now.log_prob(a_minibatch.squeeze()).view(
            -1, 1)  # shape(mini_batch_size X 1)
        # a/b=exp(log(a)-log(b))
        # shape(mini_batch_size X 1)
        ratios = torch.exp(a_logprob_now - a_logprob_minibatch)

        # Only calculate the gradient of 'a_logprob_now' in ratios
        surr1 = ratios * adv_minibatch
        surr2 = torch.clamp(ratios, 1 - self.policy_clip,
                            1 + self.policy_clip) * adv_minibatch
        actor_loss = -torch.min(surr1, surr2)
        actor_loss = actor_loss - self.entropy_coef * \
            dist_entropy  # shape(mini_batch_size X 1)
        actor_loss = actor_loss.mean()

        if guide_policy:
            auto_guide_kl_scale = abs(ratios.mean().item() -
                                      1) * guide_kl_scale
            with torch.no_grad():
                a_prob_logit = guide_policy.actor.net(s_minibatch)
                a_prob = F.softmax(a_prob_logit / temperature, dim=-1)
            action_logprob_student = F.log_softmax(logits / temperature,
                                                   dim=-1)
            kl_loss = nn.KLDivLoss(reduction='batchmean')(
                action_logprob_student, a_prob.detach()) * (temperature**2)

            loss = actor_loss + kl_loss * auto_guide_kl_scale
        else:
            auto_guide_kl_scale = 0
            kl_loss = torch.tensor(0).float().to(self.device)
            loss = actor_loss
        v_s = self.critic(s_minibatch)
        critic_loss = F.mse_loss(v_target_minibatch, v_s)
        return loss, critic_loss, kl_loss, auto_guide_kl_scale

    def update(self,
               train_steps,
               guide_policy=None,
               guide_kl_scale=1,
               temperature=1,
               use_grad_clip=True):
        s, a, a_logprob, r, s_, dw, done = self.memory.numpy_to_tensor(
        )  # Get training data
        s = s.to(self.device)
        a = a.to(self.device)
        s_ = s_.to(self.device)
        a_logprob = a_logprob.to(self.device)
        r = r.to(self.device)
        dw = dw.to(self.device)

        self.update_steps += 1
        # done=done.to(self.device)

        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = (r + self.gamma * (1.0 - dw) * vs_ - vs).cpu()
            for delta, d in zip(reversed(deltas.flatten().numpy()),
                                reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.gae_lambda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1,
                                                            1).to(self.device)
            v_target = adv + vs
            if self.config.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
        for _ in range(self.ppo_update_time):
            for index in BatchSampler(
                    SubsetRandomSampler(range(self.batch_size)),
                    self.mini_batch_size, False):

                actor_loss, critic_loss, kl_loss, kl_scale = self.calcuate_ppo_loss(
                    s[index], a[index], adv[index], a_logprob[index],
                    v_target[index], guide_policy, guide_kl_scale, temperature)

                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                # Trick 7: Gradient clip
                if use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                # Trick 7: Gradient clip
                if use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()


        return actor_loss.item(), critic_loss.item(), deltas.mean().item()


class KnowledgeExplorer(Agent):
    def __init__(self,
                 logger,
                 use_curriculum_guide=False,
                 curriculum_guide_params={},
                 use_wandb=False,
                 policy_name="PPO",
                 guide_temperature=1,
                 guide_kl_scale=1,
                 seed=0,
                 use_grad_clip=True,
                 config: config = None):
        super().__init__(logger, use_wandb, policy_name, seed, config)
        self.Policy = ExplorePolicy(cfg=self.config,
                                    logger=logger,
                                    use_wandb=use_wandb)

        self.guide_temperature = guide_temperature
        self.model = self.Policy.actor
        self.use_grad_clip = use_grad_clip
        '''
        curriculum learning settings
        '''
        self.guide_policy = None
        self.guide_policy_determinate = False
        self.guide_kl_scale = guide_kl_scale
        self.curriculum_guide_params = curriculum_guide_params
        self.use_curriculum_guide = use_curriculum_guide
        if self.use_curriculum_guide:
            assert curriculum_guide_params
        self.max_guide_episodes = self.curriculum_guide_params[
            "max_guide_episodes_rate"] * self.config.train_eps
        self.guide_step_threshold = np.linspace(
            self.curriculum_guide_params["max_guide_step_rate"],
            self.curriculum_guide_params["min_guide_step_rate"],
            int(self.max_guide_episodes))

    def reset(self):

        self.Policy.actor.reset()
        self.Policy.critic.reset()
        # self.Policy.memory.count = 0
        logging.info("Reset the model...")

    def set_guide_policy(self, guide_policy=None):
        self.guide_policy = guide_policy





class KnowledgeKeeper(Agent):
    def __init__(
            self,
            logger,
            use_wandb=False,
            policy_name="PPO",
            seed=0,
            config: config = None,
            lr=1e-3,
            kl_div_scale=1.0,
            horizion=2,
            beta=1,
            temperature=1,
            use_retrospection_loss=True,
            #  use_old_task_data=False,
            loss_metric="kd",
            compress_eval_freq=2):
        super().__init__(logger, use_wandb, policy_name, seed, config)
        self.model = self.Policy.actor
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.update_times = 0
        self.kl_div_scale = kl_div_scale
        self.old_net = copy.deepcopy(self.model.net)
        self.h = horizion
        self.beta = beta
        self.T = temperature
        self.loss_metric = loss_metric
        self.use_retrospection_loss = use_retrospection_loss
        # self.use_old_task_data = use_old_task_data
        self.old_task_transitions = Memory(Transition=expert_Transition)
        self.compress_eval_freq = compress_eval_freq

    def calculate_KL(self, train_batch, task_id):
        # print(batch[0])
        states = torch.cat([x[0] for x in train_batch])
        action_logit_teacher = torch.cat([x[1] for x in train_batch])

        action_prob_teacher = F.softmax(action_logit_teacher / self.T, dim=-1)

        action_logprob_student = F.log_softmax(self.model.net(states) / self.T,
                                               dim=-1)

        explore_loss = nn.KLDivLoss(reduction='batchmean')(
            action_logprob_student, action_prob_teacher.detach()) * (self.T**2)
        # explore_loss = F.kl_div(action_logprob_student, action_prob_teacher, size_average=False) *(self.T**2) #/ action_prob_teacher.shape[0]
        # explore_loss = F.kl_div(action_logprob_student,
        #                         action_prob_teacher.detach(),
        #                         reduction="batchmean") * (self.T**2)

        # self.logger.add_scalar("policy_preservation/explore_loss",
        #                        explore_loss.item(), self.pd_update_times)

        return explore_loss * self.kl_div_scale

    def calculate_retrospection(self, train_batch, task_id):
        # print(batch[0])

        if task_id == 0 or (not self.use_retrospection_loss):
            retrospection_loss = torch.tensor(0).float().to(self.Policy.device)
        else:
            states = torch.cat([x[0] for x in train_batch])
            with torch.no_grad():
                old_action_prob_student = F.softmax(self.old_net(states) /
                                                    self.T,
                                                    dim=-1)

            action_logprob_student = F.log_softmax(self.model.net(states) /
                                                   self.T,
                                                   dim=-1)
            retrospection_loss = nn.KLDivLoss(reduction='batchmean')(
                action_logprob_student,
                old_action_prob_student.detach()) * (self.T**2)
            # retrospection_loss = F.kl_div(action_logprob_student,
            #                     old_action_prob_student.detach(),
            #                     reduction="batchmean") * (self.T**2)
            # retrospection_loss = F.kl_div(action_logprob_student, old_action_prob_student, size_average=False) *(self.T**2) #/ old_action_prob_student.shape[0]

        return retrospection_loss * self.beta

   
    

    def compress(self,
                 all_task,
                 task_id,
                 expert_data,
                 training_batch_size,
                 update_times,
                 ewc,
                 update_early_terminate=False,
                 verbose=False):
        task = all_task[task_id]
        e_i_r = 0
        e_c_r = 0
        e_p_sr = 0
        # self.old_net = self.model.net
        with tqdm(
                range(int(update_times)),
                # leave=False,
                desc=
                f"{color.color_str(f'Compressing task {task_id}',c=color.CYAN)}"
        ) as pbar:

            for i in pbar:
                batch = random.sample(expert_data.memory, training_batch_size)

                transfer_loss = self.calculate_KL(train_batch=batch,
                                                      task_id=task_id)

                retrospection_loss = self.calculate_retrospection(
                    train_batch=batch, task_id=task_id)

                # ratio = 1 - i / int(update_times)
                kd_loss = transfer_loss + retrospection_loss  #  * (1 - ratio)

                if self.h > 0:
                    if i % self.h == 0:
                        for param, target_param in zip(
                                self.model.net.parameters(),
                                self.old_net.parameters()):
                            target_param.data.copy_(param.data)

                ewc_loss = ewc.before_backward(model=self.model,
                                               task_id=task_id)
                # nll_loss=self.NLL(train_batch=batch, task_id=task_id)
                loss = ewc_loss + kd_loss
                # loss =  ewc_loss + self.kl_div_scale *nll_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # pbar.update(1)
                self.update_times += 1



                if self.update_times % self.compress_eval_freq == 0:
                    attack_path, _, total_tasks_sr = self.Evaluate(
                        target_list=all_task[:task_id + 1],
                        verbose=False,
                        step_limit=self.config.eval_step_limit)
                    e_i_r = attack_path[0]["reward"]
                    e_c_r = attack_path[-1]["reward"]
                    e_p_sr = total_tasks_sr
                pbar.set_postfix(
                    loss_ewc=color.color_str(f"{ewc_loss.item()}",
                                             c=color.PURPLE),
                    loss_trans=color.color_str(f"{transfer_loss.item()}",
                                               c=color.YELLOW),
                    loss_kd=color.color_str(f"{kd_loss.item()}",
                                            c=color.YELLOW),
                    loss_res=color.color_str(f"{retrospection_loss.item()}",
                                             c=color.YELLOW),
                    loss=color.color_str(f"{loss.item()}", c=color.RED),
                    e_i_r=color.color_str(f"{e_i_r}", c=color.BLUE),
                    e_p_sr=color.color_str(f"{e_p_sr*100}%", c=color.CYAN),
                    e_c_r=color.color_str(f"{e_c_r}", c=color.DARKCYAN),
                )

                if e_p_sr > 0.99 and update_early_terminate:
                    break

class ScriptAgent():
    def __init__(self,
                 logger: SummaryWriter,
                 use_wandb=False,
                 policy_name="PPO",
                 seed=0,
                 config=None,
                 cl_config=None):
        # super().__init__(policy, seed, config)
        if not cl_config:
            self.cl_config = Script_Config()
        else:
            self.cl_config = cl_config
        self.policy_config = config
        self.logger = logger
        self.policy_name = policy_name
        self.name = self.cl_config.name
        self.seed = seed
        self.use_wandb = use_wandb
        self.explorer = KnowledgeExplorer(
            logger=logger,
            curriculum_guide_params=self.cl_config.curriculum_guide_params,
            use_wandb=self.use_wandb,
            policy_name=policy_name,
            seed=seed,
            config=config,
            guide_temperature=self.cl_config.guide_temperature,
            guide_kl_scale=self.cl_config.guide_kl_scale,
            use_curriculum_guide=self.cl_config.use_curriculum_guide,
            use_grad_clip=self.cl_config.use_grad_clip)
        self.keeper = KnowledgeKeeper(
            policy_name=policy_name,
            logger=logger,
            use_wandb=self.use_wandb,
            seed=seed,
            config=config,
            compress_eval_freq=self.cl_config.compress_eval_freq,
            lr=self.cl_config.pd_lr,
            kl_div_scale=self.cl_config.kl_div_scale,
            horizion=self.cl_config.horizion,
            beta=self.cl_config.beta,
            temperature=self.cl_config.temperature,
            use_retrospection_loss=self.cl_config.use_retrospection_loss,
            # use_old_task_data=self.cl_config.use_old_task_data,
            loss_metric=self.cl_config.loss_metric)

        self.config = self.explorer.config
        self.task_id = -1
        self.ewc = OnlineEWC(ewc_config=self.cl_config,
                             device=self.keeper.Policy.device)
        # self.old_task_cap = self.cl_config.retrospection_task_batch * self.cl_config.retrospection_task_num
        self.use_curriculum_guide = self.cl_config.use_curriculum_guide

    def get_new_task_learner(self, new_task_id):
        self.task_id = new_task_id
        if new_task_id != 0:
            if self.cl_config.reset_teacher:
                self.explorer.reset()
            if self.cl_config.use_keeper_init:
                for param, target_param in zip(
                        self.keeper.model.net.parameters(),
                        self.explorer.model.net.parameters()):
                    target_param.data.copy_(param.data)
            if self.use_curriculum_guide:
                self.explorer.set_guide_policy(guide_policy=self.keeper.Policy)
        return self.explorer

    def save(self, path):
        self.keeper.save(path)

    def policy_preservation(self, all_task, verbose=False):
        task = all_task[self.task_id]
        if self.explorer.use_state_norm:
            self.keeper.state_norm.running_ms.mean = self.explorer.state_norm.running_ms.mean
            self.keeper.state_norm.running_ms.std = self.explorer.state_norm.running_ms.std

        pd_transitions = self.explorer.get_expert_samples(
            target=task,
            batch_size=self.cl_config.sample_batch,
            determinate=self.cl_config.sample_determinate)

        self.keeper.model.train()
        self.keeper.compress(
            all_task=all_task,
            task_id=self.task_id,
            expert_data=pd_transitions,
            ewc=self.ewc,
            training_batch_size=self.cl_config.training_batch_size,
            update_times=self.cl_config.pd_update_times,
            update_early_terminate=self.cl_config.update_early_terminate)
        self.keeper.model.eval()

        self.ewc.after_training_task(
            task_id=self.task_id,
            agent=self.keeper,
            expert_data=pd_transitions,
            training_batch_size=self.cl_config.ewc_batch_size)


        if self.cl_config.horizion <= 0:
            for param, target_param in zip(self.keeper.model.net.parameters(),
                                           self.keeper.old_net.parameters()):
                target_param.data.copy_(param.data)

        # for (name, param) in self.keeper.Policy.actor.named_parameters():
        #     param.requires_grad = False

    def get_task_evaluator(
        self,
        on_train=False,
    ):
        if on_train:
            # 训练过程中的评估，此时还未进行策略整合
            return self.explorer
        else:
            # 训练结束后评估
            return self.keeper
