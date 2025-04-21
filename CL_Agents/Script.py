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

expert_Transition = namedtuple('pd_Transition',
                               ('state', 'action_prob', 'action', 'done'))


def gather_samples(player: Agent, target: HOST, batch_size, determinate=True):
    memory = Memory(
        Transition=namedtuple('pd_Transition', ('state', 'action_prob',
                                                'action', 'done')))
    data_num = 0
    episode = 0
    with tqdm(range(int(batch_size)),
              leave=False,
              desc=f"{color.color_str('Generate samples',c=color.CYAN)}"
              ) as pbar:
        while data_num < batch_size:
            # for data_num in pbar:
            steps = 0
            done = 0
            episode += 1
            episode_return = 0
            o = target.reset()
            if player.use_state_norm:
                o = player.state_norm(o, update=False)
            while not done and steps < player.config.step_limit:
                state = torch.tensor([o], dtype=torch.float).to(
                    player.Policy.device)
                with torch.no_grad():
                    a_logit = player.Policy.actor.net(state)
                    a_prob = F.softmax(a_logit, dim=-1)
                if not determinate:
                    dist = Categorical(probs=a_prob)
                    action = torch.squeeze(dist.sample())
                    a = int(action.item())
                else:
                    action = a_prob.argmax()
                    a = int(action)

                next_o, r, done, result = target.perform_action(a)
                episode_return += r
                if player.use_state_norm:
                    next_o = player.state_norm(next_o, update=False)
                o = next_o
                steps += 1

                memory.push(state, a_logit, action.unsqueeze(0), done)
                data_num += 1

                if data_num >= batch_size:
                    break
            pbar.update(steps)
            pbar.set_postfix(
                r=color.color_str(f"{episode_return}", c=color.PURPLE),
                step=color.color_str(f"{steps}", c=color.GREEN),
            )
    return memory


class ExplorePolicy(PPO_agent):

    def __init__(self, cfg: PPO_Config, logger: SummaryWriter, use_wandb):
        super().__init__(cfg, logger, use_wandb)
        self.AEI = 0
        self.last_delta = 0
        self.entropy_coef = 0.01

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
            return True
        return False

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

        if guide_policy and guide_kl_scale > 0:
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
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
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
                # if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                # Trick 7: Gradient clip
                if use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(),
                                                   0.5)
                self.critic_optimizer.step()
        # actor_loss_, critic_loss_, kl_loss_, ratios_ = self.calcuate_ppo_loss(
        #     s[index], a[index], adv[index], a_logprob[index], v_target[index],
        #     guide_policy, guide_kl_scale)
        # actor_loss_gain = actor_loss_ - actor_loss
        # critic_loss_gain = critic_loss_ - critic_loss
        # kl_loss_gain = kl_loss_ - kl_loss
        if self.tf_logger:
            self.tf_logger.add_scalar("loss/actor_loss", actor_loss.item(),
                                      train_steps)
            self.tf_logger.add_scalar("loss/critic_loss", critic_loss.item(),
                                      train_steps)
        if self.use_wandb:
            wandb.log({
                "loss/actor_loss": actor_loss.item(),
                # "loss/critic_loss": critic_loss.item(),
                "loss/guide_kl_scale": kl_scale,
                # "loss/actor_loss_gain": actor_loss_gain.item(),
                # "loss/critic_loss_gain": critic_loss_gain.item(),
                # "loss/kl_loss_gain": kl_loss_gain.item(),
                # "loss/delta": deltas.mean().item(),
                "loss/kl_loss": kl_loss.item(),
                "step/ppo_update": self.update_steps
            })
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

    def get_guide_action(self, observation):
        state = torch.tensor([observation],
                             dtype=torch.float).to(self.guide_policy.device)
        with torch.no_grad():
            dist = Categorical(probs=self.guide_policy.actor(state))

            sample_guide_action = dist.sample()
            guide_probs = torch.squeeze(
                dist.log_prob(sample_guide_action)).item()
            guide_action = torch.squeeze(sample_guide_action).item()

            # dist = Categorical(probs=self.Policy.actor(state))
            # sample_action = dist.sample()
            # probs = torch.squeeze(dist.log_prob(sample_action)).item()
            # probs_ = torch.squeeze(dist.log_prob(sample_guide_action)).item()
            # action = torch.squeeze(sample_action).item()
        return (guide_action, guide_probs)

    def get_expert_samples(self, target: HOST, batch_size, determinate=True):

        memory = Memory(Transition=expert_Transition)
        data_num = 0
        episode = 0
        # while data_num < batch_size:

        with tqdm(
                range(int(batch_size)),
                leave=False,
                desc=f"{color.color_str('Generate expert samples',c=color.CYAN)}"
        ) as pbar:
            while data_num < batch_size:
                # for data_num in pbar:
                steps = 0
                done = 0
                episode += 1
                episode_return = 0
                o = target.reset()
                if self.use_state_norm:
                    o = self.state_norm(o, update=False)
                while not done and steps < self.config.step_limit:
                    state = torch.tensor([o], dtype=torch.float).to(
                        self.Policy.device)
                    with torch.no_grad():
                        a_logit = self.Policy.actor.net(state)
                        a_prob = F.softmax(a_logit, dim=-1)
                    if not determinate:
                        dist = Categorical(probs=a_prob)
                        action = torch.squeeze(dist.sample())
                        a = int(action.item())
                    else:
                        action = a_prob.argmax()
                        a = int(action)

                    next_o, r, done, result = target.perform_action(a)
                    episode_return += r
                    if self.use_state_norm:
                        next_o = self.state_norm(next_o, update=False)
                    o = next_o
                    steps += 1

                    memory.push(state, a_logit, action.unsqueeze(0), done)
                    data_num += 1

                    if data_num >= batch_size:
                        break
                pbar.update(steps)
                pbar.set_postfix(
                    r=color.color_str(f"{episode_return}", c=color.PURPLE),
                    step=color.color_str(f"{steps}", c=color.GREEN),
                )
        return memory

    def run_train_episode(self, target_list, explore=False, update_norm=True):

        eps_steps = 0
        episode_return = 0
        self.action_set = []
        self.reward_set = []
        success_num = 0
        failed_num = 0
        target_id = 0
        self.task_num_episodes += 1

        if self.use_curriculum_guide and self.task_num_episodes < self.max_guide_episodes:  #and self.last_episode_reward < 0:
            max_guide_eps_steps_rate = self.guide_step_threshold[
                self.task_num_episodes]

        else:
            max_guide_eps_steps_rate = self.curriculum_guide_params[
                "min_guide_step_rate"]

            self.guide_policy = None

        max_guide_eps_steps = max_guide_eps_steps_rate * self.config.step_limit

        if self.use_reward_scaling:
            self.reward_scaling.reset()
        # for target_id in range(len(self.target_list)):
        # random.shuffle(target_list)
        while target_id < len(target_list):
            done = 0
            target_step = 0
            target: HOST = target_list[target_id]
            '''
            Init observation
            '''
            o = target.reset()
            if self.use_state_norm:
                o = self.state_norm(o, update=update_norm)

            while not done:

                if target_step >= self.config.step_limit:
                    break
                '''
                Output an action
                '''

                if self.use_curriculum_guide and self.guide_policy and eps_steps < max_guide_eps_steps:
                    action_info = self.get_guide_action(observation=o)
                else:
                    action_info = self.Policy.select_action(
                        observation=o,
                        explore=explore,
                        is_loaded_agent=self.is_loaded_agent,
                        num_episode=self.num_episodes)
                a = action_info[0]
                self.action_set.append(a)
                '''
                Perform the action
                '''
                next_o, r, done, result = target.perform_action(a)
                eps_steps += 1
                target_step += 1
                episode_return += r
                self.reward_set.append(r)
                '''
                Store the transition
                '''
                if done:
                    success_num += 1
                    dw = True
                    if self.first_hit_step < 0:
                        self.first_hit_step = self.total_training_step
                    if self.first_hit_eps < 0:
                        self.first_hit_eps = self.task_num_episodes

                else:
                    dw = False
                self.convergence_judge_done_list[
                    (self.task_num_episodes - 1) %
                    self.convergence_judge_done_num] = dw
                if self.hit_convergence_gap_eps < 0:
                    if all(self.convergence_judge_done_list):
                        self.hit_convergence_gap_eps = self.task_num_episodes - self.first_hit_eps
                        self.convergence_eps = self.task_num_episodes
                if self.use_state_norm:
                    next_o = self.state_norm(next_o, update=update_norm)
                if self.use_reward_scaling:
                    r = self.reward_scaling(r)[0]
                self.Policy.store_transtion(observation=o,
                                            action=action_info,
                                            reward=r,
                                            next_observation=next_o,
                                            done=dw)
                '''
                Update the policy
                '''
                if not explore:
                    self.total_training_step += 1

                    self.Policy.update_policy(
                        num_episode=self.num_episodes,
                        train_steps=self.total_training_step,
                        guide_policy=self.guide_policy,
                        guide_kl_scale=self.guide_kl_scale,
                        temperature=self.guide_temperature,
                        use_grad_clip=self.use_grad_clip)
                    if self.use_lr_decay:
                        #NOTE Only support PPO
                        rate = (
                            1 - self.task_num_episodes / self.config.train_eps
                        ) if self.task_num_episodes < self.config.train_eps else 1
                        if rate <= self.config.min_decay_lr:
                            rate = self.config.min_decay_lr
                        self.Policy.lr_decay(rate=rate)
                o = next_o
            if not done:
                failed_num += 1
                # break
            target_id += 1
        sucess_rate = float(format(success_num / len(target_list), '.3f'))
        if episode_return >= self.best_return:
            self.best_return = episode_return
            self.best_action_set = self.action_set
            self.best_reward_episode = self.reward_set
            self.best_episode = self.num_episodes

        return episode_return, eps_steps, sucess_rate


class KnowledgeKeeper(Agent):

    def __init__(self,
                 logger,
                 use_wandb=False,
                 policy_name="PPO",
                 seed=0,
                 config: config = None,
                 lr=1e-3,
                 transfer_strength=1.0,
                 horizion=2,
                 beta=1,
                 temperature=1,
                 use_retrospection_loss=True,
                 compress_eval_freq=2):
        super().__init__(logger, use_wandb, policy_name, seed, config)
        self.model = self.Policy.actor
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.update_times = 0
        self.transfer_strength = transfer_strength
        self.old_net = copy.deepcopy(self.model.net)
        self.h = horizion
        self.beta = beta
        self.T = temperature
        self.use_retrospection_loss = use_retrospection_loss
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

        return explore_loss * self.transfer_strength

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

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # pbar.update(1)
                self.update_times += 1

                # attack_path, _, total_tasks_sr = self.Evaluate(
                #     target_list=[all_task[task_id]],
                #     verbose=False,
                #     step_limit=self.config.eval_step_limit)
                # e_c_r = attack_path[-1]["reward"]
                # attack_path, _, total_tasks_sr = self.Evaluate(
                #     target_list=[all_task[0]],
                #     verbose=False,
                #     step_limit=self.config.eval_step_limit)
                # e_i_r = attack_path[0]["reward"]

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
                # if self.tf_logger:
                #     self.tf_logger.add_scalar("policy_preservation/loss",
                #                               loss.item(), self.update_times)
                #     self.tf_logger.add_scalar("policy_preservation/ewc_loss",
                #                               ewc_loss.item(),
                #                               self.update_times)
                #     self.tf_logger.add_scalar(
                #         "policy_preservation/retrospection_loss",
                #         retrospection_loss.item(), self.update_times)
                #     self.tf_logger.add_scalar(
                #         "policy_preservation/EvalRewards_current_task", e_c_r,
                #         self.update_times)
                #     self.tf_logger.add_scalar(
                #         "policy_preservation/EvalRewards_first_task", e_i_r,
                #         self.update_times)
                #     self.tf_logger.add_scalar(
                #         "policy_preservation/EvalSR_alltask", e_p_sr,
                #         self.update_times)

                # if self.use_wandb:
                #     wandb.log({
                #         "policy_preservation/loss":
                #         loss.item(),
                #         "policy_preservation/transfer_loss":
                #         transfer_loss.item(),
                #         "policy_preservation/ewc_loss":
                #         ewc_loss.item(),
                #         "policy_preservation/retrospection_loss":
                #         retrospection_loss.item(),
                #         "policy_preservation/EvalRewards_current_task":
                #         e_c_r,
                #         "policy_preservation/EvalRewards_first_task":
                #         e_i_r,
                #         # "policy_preservation/EvalSR_alltask":
                #         # e_p_sr,
                #         "policy_preservation/update_times":
                #         self.update_times,
                #         "policy_preservation/update_times_per_task":
                #         self.update_times / update_times
                #     })
                if e_p_sr > 0.99 and update_early_terminate:
                    break


class OnlineEWC():

    def __init__(self, ewc_config: Script_Config, device, mode="online"):
        '''
        :param ewc_lambda: hyperparameter to weigh the penalty inside the
                    total loss. The larger the lambda, the larger the
                    regularization.
        :param fisher_update_steps: How many times batches are sampled from
                    the ReplayMemory during computation of the Fisher
                    importance. Defaults to 10.
        '''
        self.ewc_config = ewc_config
        self.ewc_lambda = self.ewc_config.ewc_lambda  # "As the scale of the losses differ, we selected λ for online EWC as applied in P&C among [25, 75, 125, 175]."
        self.gamma = self.ewc_config.ewc_gamma
        self.mode = mode
        self.device = device
        self.fisher_updates_per_step = self.ewc_config.fisher_updates_per_step
        self.normaliza_fisher = self.ewc_config.normaliza_fisher
        self.saved_params: Dict[int, Dict[str, ParamData]] = defaultdict(dict)
        self.importances: Dict[int, Dict[str, ParamData]] = defaultdict(dict)
        self.ewc_use_retrospection_loss = self.ewc_config.ewc_use_retrospection_loss

    def before_backward(self, model: nn.modules, task_id):
        """
        Compute EWC penalty.
        """
        penalty = torch.tensor(0).float().to(self.device)

        if task_id > 0:  # may need importance and param expansion
            prev_exp = task_id - 1
            for k, cur_param in model.named_parameters():
                # new parameters do not count
                if k not in self.saved_params[prev_exp]:
                    continue
                saved_param = self.saved_params[prev_exp][k]
                imp = self.importances[prev_exp][k]
                new_shape = cur_param.shape
                old_imp = imp.expand(new_shape)
                old_model_para = saved_param.expand(new_shape)
                penalty += (old_imp *
                            (cur_param - old_model_para).pow(2)).sum()

        ewc_loss = self.ewc_lambda * penalty
        return ewc_loss

    def after_training_task(self, task_id, agent: KnowledgeKeeper, expert_data,
                            training_batch_size):
        """
        Compute importances of parameters after each experience.
        """
        # compute fisher information on task switch
        importances = self.compute_importances(
            agent=agent,
            expert_data=expert_data,
            training_batch_size=training_batch_size,
            optimizer=agent.optimizer,
            task_id=task_id)

        self.update_importances(importances=importances, t=task_id)

        self.saved_params[task_id] = copy_params_dict(agent.model)
        # clear previuos parameter values
        if task_id > 0:
            del self.saved_params[task_id - 1]

    def compute_importances(self, agent: KnowledgeKeeper, expert_data, task_id,
                            training_batch_size, optimizer):

        # print("Computing Importances")

        # compute importances sampling minibatches from a replay memory/buffer
        # model.train()
        importances = zerolike_params_dict(agent.model)

        for _ in range(self.fisher_updates_per_step):
            batch = random.sample(expert_data.memory, training_batch_size)

            transfer_loss = agent.calculate_KL(train_batch=batch,
                                               task_id=task_id)

            # if agent.use_old_task_data:
            #     if task_id > 0:
            #         batch = random.sample(agent.old_task_transitions.memory,
            #                               training_batch_size)
            if self.ewc_use_retrospection_loss:
                retrospection_loss = agent.calculate_retrospection(
                    train_batch=batch, task_id=task_id)
                loss = transfer_loss + retrospection_loss
            else:
                loss = transfer_loss

            # loss = agent.calculate_dkd(train_batch=batch, task_id=task_id)
            # nllloss=agent.NLL(train_batch=batch, task_id=task_id)
            # loss = explore_loss + retrospection_loss  #NOTE 此处只用explore loss是否会更好
            # loss = nllloss
            optimizer.zero_grad()
            loss.backward()

            # print(model.named_parameters(), importances)
            for (k1, p), (k2, imp) in zip(agent.model.named_parameters(),
                                          importances.items()):
                assert (k1 == k2)
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        # average over number of batches
        for _, imp in importances.items():
            imp.data /= float(self.fisher_updates_per_step)

        non_normalized_fisher_importance = copy.deepcopy(importances)
        if self.normaliza_fisher:
            for _, imp in importances.items():
                v = imp.data
                imp.data /= torch.norm(v)
            # for k, imp in importances.items():
            #     normalized_fisher_importance[k].data = imp.data/torch.norm(imp.data)
            # importances=normalized_fisher_importance
        return importances

    @torch.no_grad()
    def update_importances(self, importances, t: int):
        """
        Update importance for each parameter based on the currently computed
        importances.
        """
        if self.mode == "online":
            for (k1, old_imp), (k2, curr_imp) in itertools.zip_longest(
                    self.importances[t - 1].items(),
                    importances.items(),
                    fillvalue=(None, None),
            ):
                # Add new module importances to the importances value (New head)
                if k1 is None:
                    assert k2 is not None
                    assert curr_imp is not None
                    self.importances[t][k2] = curr_imp
                    continue

                assert k1 == k2, "Error in importance computation."
                assert curr_imp is not None
                assert old_imp is not None
                assert k2 is not None

                # manage expansion of existing layers
                self.importances[t][k1] = ParamData(
                    f"imp_{k1}",
                    curr_imp.shape,
                    init_tensor=self.gamma * old_imp.expand(curr_imp.shape) +
                    curr_imp.data,
                    device=curr_imp.device,
                )
                # if self.normaliza_fisher:
                #     normalized_fisher_importance=copy.deepcopy(self.importances[t][k1])
                #     for k, imp in self.importances[t][k1].items():
                #         normalized_fisher_importance[k].data /= torch.norm(imp.data)
            # clear previous parameter importances
            if t > 0:
                del self.importances[t - 1]

        else:
            raise ValueError("Wrong EWC mode.")


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
            logger=None,
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
            logger=None,
            use_wandb=self.use_wandb,
            seed=seed,
            config=config,
            compress_eval_freq=self.cl_config.compress_eval_freq,
            lr=self.cl_config.pd_lr,
            transfer_strength=self.cl_config.transfer_strength,
            horizion=self.cl_config.horizion,
            beta=self.cl_config.beta,
            temperature=self.cl_config.temperature,
            use_retrospection_loss=self.cl_config.use_retrospection_loss,
        )

        self.config = self.explorer.config
        self.task_id = -1
        self.ewc = OnlineEWC(ewc_config=self.cl_config,
                             device=self.keeper.Policy.device)
        self.use_curriculum_guide = self.cl_config.use_curriculum_guide

    def get_new_task_learner(self, new_task_id):
        self.task_id = new_task_id
        if new_task_id != 0:
            if self.cl_config.reset_teacher:
                self.explorer.reset()

            if self.use_curriculum_guide:
                self.explorer.set_guide_policy(guide_policy=self.keeper.Policy)
        return self.explorer

    # def save(self, path):
    #     self.keeper.save(path)

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
            update_times=self.cl_config.consolidation_iteration_num,
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
