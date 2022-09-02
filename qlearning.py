from functools import partial
from collections import defaultdict

from tqdm import tqdm
import numpy as np
from envs.gridworld import FourRooms


# POLICIES
def greedy(q, state):
    flat_ind = np.random.choice(np.flatnonzero(q[state] == q[state].max()))
    return int(list(np.unravel_index(flat_ind, q[state].shape))[0])


def e_greedy(q, state, n_actions, eps) -> int:
    if np.random.rand() < eps:
        return int(np.random.randint(0, n_actions))
    else:
        return greedy(q, state)


# Create Q
def create_q(n_actions: int):
    return defaultdict(lambda: np.zeros(n_actions))


# Q-LEARNING
def qlearning(env: FourRooms,
              max_episodes=1000,
              alpha=1, gamma=1,
              eps=0.2,
              is_decaying_eps=False, start_eps=0.99, end_eps=0.01, eps_incr=None,
              ):

    n_actions = env.action_space.n
    stats = {"steps": {}, "returns": {}}

    policy = partial(e_greedy, n_actions=n_actions)
    q = create_q(n_actions)

    curr_step = 0
    curr_episode = 0

    if eps_incr is None:
        eps_incr = (start_eps - end_eps)/max_episodes

    if is_decaying_eps:
        eps = start_eps

    def stop_cond():
        return curr_episode >= max_episodes

    prog_bar = tqdm(total=max_episodes, desc="Episodes")

    while not stop_cond():
        curr_state = env.reset()
        is_done = False
        while not is_done and not stop_cond():
            action = policy(state=curr_state, q=q, eps=eps)
            next_state, reward, is_done, info = env.step(action)

            s = curr_state
            a = action
            s_next = next_state

            td_target = reward + gamma * np.max(q[s_next]) * (not is_done)
            td_error = td_target - q[s][a]

            q[s][a] = q[s][a] + alpha * td_error

            curr_state = next_state
            curr_step += 1

        stats["steps"][curr_episode] = env.episode_step
        stats["returns"][curr_episode] = env.curr_return

        prog_bar.update()
        prog_bar.set_postfix({"steps": curr_step, "last return": env.curr_return, "epsilon": eps})

        curr_episode += 1

        if is_decaying_eps:
            if eps - eps_incr >= start_eps:
                eps -= eps_incr

    return q, stats


def main():
    env = FourRooms()
    q, stats = qlearning(env, 1000)
    print(stats)


if __name__ == "__main__":
    main()
