from arguments import get_args
from ppo_agent import ppo_agent
from utils.env_wrapper.create_env import create_multiple_envs, create_single_env
from utils.seeds.seeds import set_seeds
import os

if __name__ == '__main__':
    args = get_args()
    if args.env_type == 'mujoco':
        envs = create_single_env(args)
    elif args.env_type == 'atari':
        envs = create_multiple_envs(args)
    else:
        raise NotImplementedError
    set_seeds(args)
    ppo_trainer = ppo_agent(envs, args)
    ppo_trainer.learn()
    envs.close()
