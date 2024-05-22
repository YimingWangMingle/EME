from utils.env_wrapper.atari_wrapper import make_atari, wrap_deepmind
from utils.env_wrapper.mario_wrappers import wrap_mario
from utils.env_wrapper.multi_envs_wrapper import SubprocVecEnv
from utils.env_wrapper.frame_stack import VecFrameStack
from utils.logger import logger, bench
import os
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import os
import gym
import cv2
import numpy as np
import collections



def create_single_env(args, rank=0):
    if rank == 0:
        if not os.path.exists(args.log_dir):
            os.mkdir(args.log_dir)
        log_path = args.log_dir + '/{}/'.format(args.env_name)
        logger.configure(log_path)
    if args.env_type == 'atari':
        env = make_atari(args.env_name)
        env = bench.Monitor(env, logger.get_dir())
        env = wrap_deepmind(env, frame_stack=True)
    else:
        env = gym.make(args.env_name)
        env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
    env.seed(args.seed + rank)
    return env


def create_multiple_envs(args):
    if args.env_type == 'atari':
        def make_env(rank):
            def _thunk():
                if not os.path.exists(args.log_dir):
                    os.makedirs(args.log_dir, exist_ok=True)
                log_path = args.log_dir + '/{}/'.format(args.env_name)
                logger.configure(log_path)
                env = make_atari(args.env_name)
                env.seed(args.seed + rank)
                env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
                env = wrap_deepmind(env)
                return env
            return _thunk
        envs = SubprocVecEnv([make_env(i) for i in range(args.num_workers)])
        envs = VecFrameStack(envs, 4)
    elif args.env_type == 'mario':
        def make_env(rank):
            def _thunk():
                if not os.path.exists(args.log_dir):
                    os.makedirs(args.log_dir, exist_ok=True)
                log_path = args.log_dir + '/{}/'.format(args.env_name)
                logger.configure(log_path)
                # env = gym.make(args.env_name)
                env = wrap_mario(args.env_name)
                env.seed(args.seed + rank)
                env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
                return env
            return _thunk
        envs = SubprocVecEnv([make_env(i) for i in range(args.num_workers)])
    else:
        raise NotImplementedError
    return envs

