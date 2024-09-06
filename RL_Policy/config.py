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
        self.min_decay_lr = min_decay_lr


class SAC_Config(config):
    def __init__(self,
                 batch_size=2048,
                 memory_size=None,
                 random_step=1e4,
                 gamma=0.99,
                 dim_reduction=True,
                 reduction_action_dim=30,
                 actor_lr=1e-4,
                 critic_lr=1e-4,
                 lr_alpha=5e-5,
                 tau=5e-2,
                 hidden_sizes=1024,
                 random_radio=0.01,
                 target_entropy=-10,
                 use_grad_clip=False,
                 use_orthogonal_init=False,
                 adaptive_alpha=True,
                 k_nearest_neighbors=100,
                 use_k_decay=False,
                 activate_func="leaky_relu",
                 policy_frequency=1,
                 target_network_frequency=1,
                 action_refinement="UCB",
                 use_dim_reduction=True,
                 ucb_lamba=10.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.memory_size = self.train_eps * self.step_limit if not memory_size else memory_size
        self.random_step = random_step
        self.gamma = gamma
        self.dim_reduction = dim_reduction
        self.reduction_action_dim = reduction_action_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.lr_alpha = lr_alpha
        self.tau = tau
        self.hidden_sizes = hidden_sizes
        self.random_radio = random_radio
        # self.target_entropy = -max(1.0, 0.98*math.log(self.action_dim))  #
        self.target_entropy = target_entropy
        self.use_grad_clip = use_grad_clip

        self.use_orthogonal_init = use_orthogonal_init
        self.adaptive_alpha = adaptive_alpha
        self.k_nearest_neighbors = k_nearest_neighbors
        self.use_k_decay = use_k_decay
        self.max_k = self.k_nearest_neighbors
        self.min_k = 5
        self.reward_scale_ratio = 1
        self.activate_func = activate_func
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.action_refinement = action_refinement  # Greedy   Random
        self.distance_loss_add_to = "actor"  #
        self.use_distance_loss = "ContrastiveLoss"  #    CosineEmbeddingLoss
        self.distance_loss_beta = 0.1 * self.k_nearest_neighbors
        self.use_dim_reduction = use_dim_reduction
        self.nearest_neighbor = "annoy"
        self.nn_metric = "angular"  #manhattan   euclidean

        self.delta = 0
        self.ucb_lamba = ucb_lamba
        self.chain_mode = "OneByOne"  #Sequence   OneByOne
        # pprint(locals())


class DQN_Config(config):
    def __init__(self,
                 lr=2e-4,
                 target_update_freq=200,
                 current_update_freq=1,
                 batch_size=4096,
                 replay_size=5e4,
                 final_epsilon=0.001,
                 epsilon_exploration_episode=200,
                 gamma=0.8,
                 hidden_sizes=512,
                 use_per=False,
                 explore_eps=0,
                 activate_func="relu",
                 TAU=5e-3,
                 **kwargs):
        super().__init__(**kwargs)
        self.lr = lr
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.final_epsilon = final_epsilon
        self.epsilon_exploration_episode = epsilon_exploration_episode
        self.gamma = gamma
        self.hidden_sizes = [hidden_sizes, hidden_sizes]
        self.target_update_freq = target_update_freq
        self.current_update_freq = current_update_freq
        self.use_per = use_per  #Prioritized Experience Replay
        self.explore_eps = explore_eps if not self.use_per else self.replay_size // self.step_limit  #  no Prioritized Experience Replay
        self.TAU = TAU
        self.activate_func = activate_func
        self.use_state_norm = False


class MeDQN_Config(cl_config):
    def __init__(self,
                 lamda=0,
                 max_lamda=40,
                 min_lamda=10,
                 consod_epoch=1,
                 consod_batch_size=64,
                 policy_name="DQN",
                 reset=False,
                 loss_func="huber",
                 **kwargs):
        super().__init__()
        if policy_name == "DQN":
            policy_config = DQN_Config(**kwargs)
        else:
            raise ValueError()
        for k, v in policy_config.__dict__.items():
            setattr(self, k, v)
        self.continual_learning = True
        self.lamda = lamda
        self.max_lamda = max_lamda
        self.min_lamda = min_lamda
        self.consod_epoch = consod_epoch
        self.reset = reset
        self.loss_func = loss_func
        self.consod_batch_size = consod_batch_size
        self.name = "MeDQN"


class D3QN_Config(config):
    def __init__(self,
                 lr=1e-4,
                 batch_size=128,
                 replay_size=10000,
                 final_epsilon=0.01,
                 epsilon_exploration_episode=300,
                 gamma=0.99,
                 hidden_sizes=[512, 512],
                 use_per=True,
                 explore_eps=50,
                 TAU=5e-3,
                 **kwargs):
        super().__init__(**kwargs)
        self.lr = lr
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.final_epsilon = final_epsilon
        self.epsilon_exploration_episode = epsilon_exploration_episode
        self.gamma = gamma
        self.hidden_sizes = hidden_sizes
        # self.target_update_freq =200
        self.use_per = use_per  #Prioritized Experience Replay
        self.explore_eps = explore_eps if not self.use_per else self.replay_size // self.step_limit  #  no Prioritized Experience Replay
        self.TAU = TAU
        self.activate_func = "tanh"
        self.use_state_norm = False


class HADRL_Config(config):
    def __init__(self):
        super().__init__()
        self.lr = 1e-4
        self.train_episodes = 500
        self.step_limit = 100
        self.batch_size = 2048
        self.use_state_norm = True
        self.final_epsilon = 0.01
        self.exploration_episode = 200
        self.gamma = 0.9
        self.hidden_sizes = 1024
        self.target_update_freq = 1000
        self.eval_step_limit = 10
        self.explore_episode = 50
        self.TAU = 0.05
        self.replay_size = self.step_limit * (self.train_episodes +
                                              self.explore_episode)


class Random_Config(config):
    def __init__(self, seed=0):
        super().__init__()
        self.seed = seed
        self.step_limit = 100
        self.eval_step_limit = 10


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


class PolicyConsolidation_Config(cl_config):
    def __init__(
            self,
            policy_name="PPO",
            actor_lr=5e-5,
            critic_lr=5e-5,  # 3e-4
            omega12=0.25,
            omega=1,
            beta=0.3,
            hidden_policies=5,
            policy_clip=0.3,
            entropy_coef=0.05,
            **kwargs):
        super().__init__()
        if policy_name == "PPO":
            policy_config = PPO_Config(**kwargs)
        else:
            raise ValueError()
        for k, v in policy_config.__dict__.items():
            setattr(self, k, v)
        self.continual_learning = True
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.reset = False
        self.name = "PolicyConsolidation"
        self.omega12 = omega12
        self.omega = omega
        self.beta = beta
        self.hidden_policies = hidden_policies
        self.policy_clip = policy_clip
        self.entropy_coef = entropy_coef


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


class EWC_Config(PolicyDistillation_Config):
    def __init__(self,
                 ewc_lambda=4500,
                 ewc_gamma=0.99,
                 fisher_updates_per_step=10,
                 kl_div_scale=1,
                 normaliza_fisher=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.ewc_lambda = ewc_lambda  # "As the scale of the losses differ, we selected λ for online EWC as applied in P&C among [25, 75, 125, 175]."
        self.ewc_gamma = ewc_gamma  # "γ < 1 is a hyperparameter associated with removing the approximation term associated with the previous presen-tation of task i."
        self.fisher_updates_per_step = fisher_updates_per_step
        self.kl_div_scale = kl_div_scale
        self.normaliza_fisher = normaliza_fisher


class ProgressCompress_Config(PolicyDistillation_Config):
    def __init__(self,
                 sample_batch=1000,
                 pd_update_times=200,
                 ewc_lambda=1000,
                 ewc_gamma=0.99,
                 pd_lr=5e-5,
                 training_batch_size=128,
                 fisher_updates_per_step=10,
                 kl_div_scale=1,
                 normaliza_fisher=True,
                 reset_teacher=True,
                 adapter_activate_func="tanh",
                 **kwargs):
        super().__init__(**kwargs)
        self.pd_lr = pd_lr
        self.adapter_activate_func = adapter_activate_func
        self.sample_batch = sample_batch
        self.pd_update_times = pd_update_times
        self.training_batch_size = training_batch_size
        self.ewc_lambda = ewc_lambda  # "As the scale of the losses differ, we selected λ for online EWC as applied in P&C among [25, 75, 125, 175]."
        self.ewc_gamma = ewc_gamma  # "γ < 1 is a hyperparameter associated with removing the approximation term associated with the previous presen-tation of task i."
        self.fisher_updates_per_step = fisher_updates_per_step
        self.kl_div_scale = kl_div_scale
        self.normaliza_fisher = normaliza_fisher
        self.reset_teacher = reset_teacher
        self.name = "Progress&Compress"
        self.compress_eval_freq = 100


class Script_Config(PolicyDistillation_Config):
    def __init__(
            self,
            pd_lr=5e-5,  #3
            sample_batch=5000,
            ewc_lambda=2000,  #1000
            training_batch_size=256,
            ewc_gamma=0.99,
            ewc_batch_size=128,
            fisher_updates_per_step=10,
            kl_div_scale=0.7,
            normaliza_fisher=True,
            horizion=0,
            beta=1,  #0.5
            temperature=0.5,  #
            guide_temperature=0.1,  #
            pd_update_times=1000,
            update_early_terminate=False,
            compress_eval_freq=100,
            use_retrospection_loss=True,
            ewc_use_retrospection_loss=True,
            # use_old_task_data=False,
            explorer_mode="from_scratch",
            use_curriculum_guide=True,
            guide_kl_scale=1,
            max_guide_episodes_rate=0.1,
            max_guide_step_rate=0.5,
            use_grad_clip=True,
            **kwargs):
        super().__init__(**kwargs)
        self.pd_lr = pd_lr
        self.use_grad_clip = use_grad_clip
        self.sample_batch = sample_batch
        self.training_batch_size = training_batch_size
        self.ewc_lambda = ewc_lambda  # "As the scale of the losses differ, we selected λ for online EWC as applied in P&C among [25, 75, 125, 175]."
        self.ewc_gamma = ewc_gamma  # "γ < 1 is a hyperparameter associated with removing the approximation term associated with the previous presen-tation of task i."
        self.fisher_updates_per_step = fisher_updates_per_step
        self.ewc_batch_size = training_batch_size
        self.ewc_use_retrospection_loss = use_retrospection_loss
        self.kl_div_scale = kl_div_scale
        self.normaliza_fisher = normaliza_fisher
        self.horizion = horizion
        self.beta = beta
        self.temperature = temperature
        self.guide_temperature = guide_temperature
        self.pd_update_times = pd_update_times
        self.update_early_terminate = update_early_terminate
        self.compress_eval_freq = compress_eval_freq
        self.loss_metric = "kd"
        self.use_retrospection_loss = use_retrospection_loss
        # self.use_old_task_data = use_old_task_data
        self.retrospection_task_batch = self.training_batch_size
        self.guide_kl_scale = guide_kl_scale
        self.use_curriculum_guide = use_curriculum_guide
        self.curriculum_guide_params = {
            "max_guide_episodes_rate": max_guide_episodes_rate,
            "max_guide_step_rate": max_guide_step_rate,
            "min_guide_step_rate": 0,
        }
        self.explorer_mode = explorer_mode

        if self.explorer_mode == "explorer_assignment":
            self.reset_teacher = False
            self.use_keeper_init = False
        elif self.explorer_mode == "keeper_assignment":
            self.reset_teacher = False
            self.use_keeper_init = True
        elif self.explorer_mode == "from_scratch":
            self.reset_teacher = True
            self.use_keeper_init = False
        else:
            exit(0)
        if self.use_curriculum_guide:
            self.explorer_mode += "+curriculum_guide"
        self.name = f"Script_{self.explorer_mode}"
        if not use_retrospection_loss:
            self.name += "+no_res"
        if not ewc_lambda:
            self.name += "+no_ewc"
        if not guide_kl_scale:
            self.name += "+no_imitation"


if __name__ == '__main__':

    para = {}
    # para["pd_lr"] = 3e-4
    para["gamma"] = 0.9
    cfg = Finetune_Config(**para)
    pprint(cfg)
