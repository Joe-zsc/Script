# Script: **S**calable **C**ontinual **R**e**I**nforcement learning framework for autonomous **P**enetration **T**esting

## Introduction

This repository is a simplified implementation of the paper [SCRIPT: A Scalable Continual Reinforcement Learning Framework for Autonomous Penetration Testing](https://www.sciencedirect.com/science/article/abs/pii/S0957417425014496).  In this work, we introduce SCRIPT, which is the first scalable continual reinforcement learning framework for autonomous pentesting, enabling agents to leverage previously learned knowledge to accelerate new task learning while avoiding catastrophic forgetting.

## Related Works

[GAP](https://github.com/Joe-zsc/GAP)

[April-AE](https://github.com/Joe-zsc/April-AE)

## Getting Started

### Installation

Start by checking out the repository:

```bash
git clone https://github.com/Joe-zsc/Script.git
cd Script
pip install -r requirments.txt
```

### Prepare the embedding models

In this project, we directly use sentence-bert to represent the raw state information and action descriptions as vectors.

1. Download pre-trained [Sentence-BERT](https://huggingface.co/models?library=sentence-transformers) models, or train/fine-tune your own embedding models using domain corpus. (reference: [TSDAE](https://github.com/UKPLab/sentence-transformers))
2. Store the embedding models in path  `NLP_Module\Embedding_models`.
3. Modify the config file `config.ini` and write the model names in the corresponding positions.

```ini
[Embedding]
embedding_models = NLP_Module\Embedding_models
sbert_model = MySbertModel ; your sentence-bert model name, e,g., all-MiniLM-L12-v2
```

4. Check the simulated training scenarios in `scenarios` file, which are constructed by pre-probing the vulnerable hosts in Vulhub.

## Hyperparameters of SCRIPT

Default key hyperparameter settings can be found in ``RL_policy/config.py `` : see class ``Script_Config``

### Training with simulated environments

Run the following commands to run a simulation with SCRIPT (continually training 10 tasks):

```bash
python April_continual.py --cl_method "script" --cl_train_num 10 --seed 0 
```

The learning curves can be seen via the Tensorboard:

```bash
tensorboard --logdir runs --host localhost --port 6666
```

## Citation

```
@article{zhou_script_2025,
	title = {{SCRIPT}: A Scalable Continual Reinforcement Learning Framework for Autonomous Penetration Testing},
	volume = {285},
	issn = {0957-4174},
	doi = {https://doi.org/10.1016/j.eswa.2025.127827},
	pages = {127827},
	journaltitle = {Expert Systems with Applications},
	author = {Zhou, Shicheng and Liu, Jingju and Lu, Yuliang and Yang, Jiahai and Zhang, Yue and Lin, Bo and Zhong, Xiaofeng and Hu, Shulong},
	date = {2025},
}
```
