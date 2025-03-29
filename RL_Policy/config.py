import sys, os

from pprint import pprint

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path


class config:

    def __init__(self,
                 train_eps=500,
                 explore_eps=0,
                 step_limit=100,
                 eval_step_limit=10,
                 activate_func="relu",
                 use_state_norm=True,
                 use_lr_decay=False,
                 use_reward_scaling=False):
        self.continual_learning = False
        self.train_eps = train_eps
        self.explore_eps = explore_eps
        self.step_limit = step_limit
        self.eval_step_limit = eval_step_limit
        self.activate_func = activate_func
        self.use_state_norm = use_state_norm
        self.use_lr_decay = use_lr_decay
        self.use_reward_scaling = use_reward_scaling


        # self.min_decay_lr = 3e-2
class cl_config():

    def __init__(self) -> None:
        self.continual_learning = True


class PPO_Config(config):

    def __init__(
            self,
            batch_size=512,
            mini_batch_size=64,
            gamma=0.99,
            ppo_update_time=8,
            actor_lr=1e-4,
            critic_lr=5e-5,  # 3e-4
            Adam_Optimizer_Epsilon=1e-7,
            gae_lambda=0.95,
            policy_clip=0.2,
            hidden_sizes=[512, 512],
            entropy_coef=0.02,
            activate_func="tanh",  #tanh
            use_layer_norm=False,
            use_orthogonal_init=False,
            min_decay_lr=5e-1,
            **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size  #64
        self.gamma = gamma
        self.ppo_update_time = ppo_update_time
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.Adam_Optimizer_Epsilon = Adam_Optimizer_Epsilon
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.hidden_sizes = hidden_sizes
        self.entropy_coef = entropy_coef
        self.activate_func = activate_func
        self.use_adv_norm = self.use_state_norm
        self.use_orthogonal_init = use_orthogonal_init
        self.use_layer_norm = use_layer_norm
        self.min_decay_lr = min_decay_lr


class Finetune_Config(cl_config):

    def __init__(self, policy_name="PPO", **kwargs):
        super().__init__()
        if policy_name == "PPO":
            policy_config = PPO_Config(**kwargs)
        else:
            raise ValueError()
        for k, v in policy_config.__dict__.items():
            setattr(self, k, v)
        self.continual_learning = True
        self.reset = False
        if self.reset:
            self.name = "LearnFromScratch"
        else:
            self.name = "Finetune"





class PolicyDistillation_Config(cl_config):

    def __init__(self,
                 sample_batch=5000,
                 training_batch_size=128,
                 pd_update_times=200,
                 pd_lr=5e-5,
                 reset_teacher=True,
                 sample_determinate=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.sample_batch = sample_batch
        self.training_batch_size = training_batch_size
        self.pd_update_times = pd_update_times
        self.pd_lr = pd_lr
        self.loss_metric = 'kd'
        self.reset_teacher = reset_teacher
        self.sample_determinate = sample_determinate


class Script_Config(PolicyDistillation_Config):

    def __init__(
            self,
            pd_lr=5e-5,  #3
            sample_batch=5000,
            ewc_lambda=2000,
            training_batch_size=256,
            ewc_gamma=0.99,
            fisher_updates_per_step=10,
            transfer_strength=0.7,  # for policy transfer
            normaliza_fisher=True,
            horizion=0,  # 0 means horizion is set to as consolidation_iteration_num
            beta=1,  # for policy retrospection
            temperature=0.5,  #
            guide_temperature=0.1,  # for policy imitaion
            consolidation_iteration_num=1000, # iteration number of knowledge consolidation process
            update_early_terminate=False,
            compress_eval_freq=100,
            use_retrospection_loss=True,
            reset_teacher=True,
            use_curriculum_guide=True,
            guide_kl_scale=2,  # for policy imitaion
            max_guide_episodes_rate=0.1,
            max_guide_step_rate=0.5,
            use_grad_clip=True,
            **kwargs):
        super().__init__(**kwargs)
        self.pd_lr = pd_lr
        self.use_grad_clip = use_grad_clip
        self.sample_batch = sample_batch
        self.training_batch_size = training_batch_size
        self.ewc_lambda = ewc_lambda  # "the scale of the losses differ"
        self.ewc_gamma = ewc_gamma  # "γ < 1 is a hyperparameter associated with removing the approximation term associated with the previous presen-tation of task i."
        self.fisher_updates_per_step = fisher_updates_per_step
        self.ewc_batch_size = training_batch_size
        self.ewc_use_retrospection_loss = use_retrospection_loss
        self.transfer_strength = transfer_strength
        self.normaliza_fisher = normaliza_fisher
        self.horizion = horizion
        self.beta = beta
        self.temperature = temperature
        self.guide_temperature = guide_temperature
        self.consolidation_iteration_num = consolidation_iteration_num
        self.update_early_terminate = update_early_terminate
        self.compress_eval_freq = compress_eval_freq
        self.use_retrospection_loss = use_retrospection_loss
        self.retrospection_task_batch = self.training_batch_size
        self.guide_kl_scale = guide_kl_scale
        self.use_curriculum_guide = use_curriculum_guide
        self.curriculum_guide_params = {
            "max_guide_episodes_rate": max_guide_episodes_rate,
            "max_guide_step_rate": max_guide_step_rate,
            "min_guide_step_rate": 0,
        }
        self.reset_teacher = reset_teacher
        self.name = "Script"

        if not self.use_curriculum_guide:
            self.name += "+no_curriculum_guide"
        if not self.reset_teacher:
            self.name += "no_model_reinit"

        if not use_retrospection_loss:
            self.name += "+no_res"
        if ewc_lambda <= 0:
            self.name += "+no_ewc"
        if guide_kl_scale <= 0:
            self.name += "+no_imitation"


if __name__ == '__main__':

    para = {}
    # para["pd_lr"] = 3e-4
    para["gamma"] = 0.9
    cfg = Finetune_Config(**para)
    pprint(cfg)
