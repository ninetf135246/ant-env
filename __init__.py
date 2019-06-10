"""Random policy on an environment."""

import tensorflow as tf
import numpy as np
import random
import argparse

import create_maze_env
import imageio as io

def get_goal_sample_fn(env_name):
    if env_name == 'AntMaze':
        # NOTE: When evaluating (i.e. the metrics shown in the paper,
        # we use the commented out goal sampling function.    The uncommented
        # one is only used for training.
        #return lambda: np.array([0., 16.])
        return lambda: np.random.uniform((-4, -4), (20, 20))
    elif env_name == 'AntPush':
        return lambda: np.array([0., 19.])
    elif env_name == 'AntFall':
        return lambda: np.array([0., 27., 4.5])
    else:
        assert False, 'Unknown env'


def get_reward_fn(env_name):
    if env_name == 'AntMaze':
        return lambda obs, goal: -np.sum(np.square(obs[:2] - goal)) ** 0.5
    elif env_name == 'AntPush':
        return lambda obs, goal: -np.sum(np.square(obs[:2] - goal)) ** 0.5
    elif env_name == 'AntFall':
        return lambda obs, goal: -np.sum(np.square(obs[:3] - goal)) ** 0.5
    else:
        assert False, 'Unknown env'


def success_fn(last_reward):
    return last_reward > -5.0


class EnvWithGoal(object):
    def __init__(self, base_env, env_name):
        self.base_env = base_env
        self.goal_sample_fn = get_goal_sample_fn(env_name)
        self.reward_fn = get_reward_fn(env_name)
        self.goal = None

    def reset(self):
        # self.viewer_setup()
        obs = self.base_env.reset()
        self.goal = self.goal_sample_fn()
        return np.concatenate([obs, self.goal])

    def step(self, a):
        obs, _, done, info = self.base_env.step(a)
        reward = self.reward_fn(obs, self.goal)
        return np.concatenate([obs, self.goal]), reward, done, info

    def render(self):
        return self.base_env.render()

    def get_image(self):
        img_data = self.render()
        #print(img_data.shape)
        #data = self.base_env.viewer.get_image()

        #img_data = data[0]
        #width = data.shape[0]
        #height = data.shape[1]

        #tmp = np.fromstring(img_data, dtype=np.uint8)
        #image_obs = np.reshape(tmp, [height, width, 3])
        img_data = np.flipud(img_data)

        #return image_obs
        return img_data

    @property
    def action_space(self):
        return self.base_env.action_space


def run_environment(env_name, episode_length, num_episodes, render_size):
    env = EnvWithGoal(
            create_maze_env.create_maze_env(env_name, render_size=render_size),
            env_name)

    def action_fn(obs):
        action_space = env.action_space
        action_space_mean = (action_space.low + action_space.high) / 2.0
        action_space_magn = (action_space.high - action_space.low) / 2.0
        random_action = (action_space_mean +
            action_space_magn *
            np.random.uniform(low=-1.0, high=1.0,
            size=action_space.shape))

        return random_action

    rewards = []
    successes = []
    all_images = []
    for ep in range(num_episodes):
        rewards.append(0.0)
        successes.append(False)
        obs = env.reset()
        for _ in range(episode_length):
            env.render()
            #print(env.get_image().shape)
            obs, reward, done, _ = env.step(action_fn(obs))
            rewards[-1] += reward
            successes[-1] = success_fn(reward)
            all_images.append(env.get_image())
            if done:
                break

        print('Episode {} reward: {}, Success: {}'.format(ep + 1, rewards[-1], successes[-1]))
    
    all_images = np.array(all_images)
    print(all_images.shape)
    io.mimsave('imgs.gif', all_images)
    print('Average Reward over {} episodes: {}'.format(num_episodes, np.mean(rewards)))
    print('Average Success over {} episodes: {}'.format(num_episodes, np.mean(successes)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="AntMaze", type=str)               
    parser.add_argument("--ep-len", default=500, type=int)      
    parser.add_argument("--num-ep", default=100, type=int)
    parser.add_argument("--render-h", default=200, type=int)
    parser.add_argument("--render-w", default=200, type=int)

    args = parser.parse_args()
    run_environment(args.env_name, args.ep_len, args.num_ep, (args.render_w, args.render_h))
