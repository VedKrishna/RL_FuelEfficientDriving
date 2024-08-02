import sys
import gym
import time
import numpy as np
import pandas as pd

def obs_to_state(env, obs):
    """Maps an observation to state"""
    env_low = env.observation_space.low
    env_high = env.observation_space.high

    env_dx = (env_high - env_low) / n_states
    # print(env_low)
    # print(env_dx)

    a = int((obs[0] - env_low[0])/env_dx[0])
    b = int((obs[1] - env_low[1])/env_dx[1])

    return a,b

def run_episode(env, policy = None, render = False):

    obs = env.reset()
    total_reward = 0
    step_idx = 0

    a,b = obs_to_state(env, obs[0])
    flag = False

    for _ in range (t_max):
        if render:
            env.render()

        if policy is None:
            action = env.action_space.sample()
        else:
            if flag:
                a,b = obs_to_state(env, obs)
            else:
                flag = True
            action = policy[a][b]

        obs, reward, done, _, __ = env.step(action)
        total_reward += gamma ** step_idx * reward

        step_idx += 1

        if done:
            break

    return total_reward

env_name = 'CarEnvironment'

env = gym.make(env_name, render_mode = 'human')
np.random.seed(0)

n_states = 200
iter_max = 20000
t_max = 1000

initial_lr = 1
min_lr = 0.003
gamma = 1.0
eps = 0.05

print(' using Q Learning -----')

q_table = np.zeros((n_states+1, n_states+1, 3))

for i in range(iter_max):

    obs = env.reset()
    total_reward = 0

    ## eta: learning rate is decreased at each step
    eta = max(min_lr, initial_lr * (0.85** (i//100)))
    a, b = obs_to_state(env, obs[0])

    for j in range(t_max):

        if np.random. uniform (0, 1) < eps:
            action = np.random.choice(env.action_space.n)
        else:
            logits = q_table[a][b]
            max_logit = np.max(logits)
            stabilized_logits = logits - max_logit
            exps = np.exp(stabilized_logits)
            probs = exps / np.sum(exps)
            probs /= np.sum(probs)
            action = np.random.choice(env.action_space.n, p=probs)

        obs, reward, done, _, __ = env.step(action)
        total_reward += reward

        a, b = obs_to_state(env, obs)

        # update q table
        a_, b_ = obs_to_state(env, obs)
        if __['fuel'] < 0:
            q_table [a][b][0] = 0
            q_table [a][b][1] = 1
            q_table [a][b][2] = 0
        else:
            q_table [a][b][action] = round(q_table [a][b][action] + eta * (reward + gamma * np.max(q_table [a_] [b_]) - q_table [a] [b] [action]), 3)

        if done:
            break

    if i % 1000 == 0:

        print('Iteration #%d -- Total reward = %d.' %(i+1, total_reward))

solution_policy = np.argmax(q_table, axis=2)
solution_policy_scores = [run_episode(env, solution_policy, False) for _ in range(100)]
print("Average score of solution = ", np.mean (solution_policy_scores))

# Animate it

print(solution_policy)
run_episode(env, solution_policy, True)