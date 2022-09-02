from functools import partial
from collections import defaultdict

from tqdm import tqdm
import numpy as np
from envs.gridworld import FourRooms


# POLICIES
def greedy(q, state, n_actions, goal=None):
    if goal is not None:
        flat_ind = np.random.choice(np.flatnonzero(q[state][goal] == q[state][goal].max()))
        return int(list(np.unravel_index(flat_ind, q[state][goal].shape))[0])
    else:
        q_arrs = [q[state][g] for g in q[state].keys()]

        if len(q_arrs) == 0:
            qvf = np.zeros(n_actions)
        else:
            qvf = np.max(q_arrs, axis=0)

        # take argmax on QVF
        flat_ind = np.random.choice(np.flatnonzero(qvf == qvf.max()))
        action = int(list(np.unravel_index(flat_ind, qvf.shape))[0])
        return action


def e_greedy(q, state, n_actions, eps, goal=None) -> int:
    if np.random.rand() < eps:
        return int(np.random.randint(0, n_actions))
    else:
        return greedy(q, state, n_actions, goal)


# Create Q
def create_q(n_actions: int):
    return defaultdict(lambda: defaultdict(lambda: np.zeros(n_actions)))


# Goal Oriented Learning
def golearning(env: FourRooms,
               max_episodes=1000,
               alpha=1, gamma=1,
               eps=0.2,
               is_decaying_eps=False, start_eps=0.99, end_eps=0.01, eps_incr=None,
              ):

    n_actions = env.action_space.n
    stats = {"steps": {}, "returns": {}}
    goal_library = {}

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
    goal_desirable = None

    while not stop_cond():
        if len(goal_library) == 0:
            goal = None
        else:
            goals_arr = list(goal_library.keys())
            goal_ind = np.random.randint(len(goals_arr))
            goal = goals_arr[goal_ind]

        curr_state = env.reset()
        is_done = False
        while not is_done and not stop_cond():
            if goal is None:
                action = policy(state=curr_state, q=q, eps=1)
                goal_desirable = None
            else:
                action = policy(state=curr_state, q=q, eps=eps, goal=goal)
                goal_desirable = goal_library[goal]

            next_state, reward, is_done, info = env.step(action)

            if is_done:
                # Goal Oriented Learning does not require storing info on if the goal in the goal_library
                # is desirable or undesirable. This is stored for easier logging & debugging
                if reward > 0:
                    goal_library[curr_state] = True
                else:
                    goal_library[curr_state] = False

            for learn_goal in goal_library.keys():
                if curr_state in goal_library and curr_state != learn_goal:
                    r_learn = env.rmin
                else:
                    r_learn = reward

                s = curr_state
                a = action
                g_prime = learn_goal
                s_next = next_state

                td_target = r_learn + gamma * np.max(q[s_next][g_prime]) * (not is_done)
                td_error = td_target - q[s][g_prime][a]

                q[s][g_prime][a] = q[s][g_prime][a] + alpha * td_error

            curr_state = next_state
            curr_step += 1

        stats["steps"][curr_episode] = env.episode_step
        stats["returns"][curr_episode] = env.curr_return

        prog_bar.update()
        prog_bar.set_postfix({"steps": curr_step, "last return": env.curr_return,
                              "epsilon": eps, "goal desirable": goal_desirable})

        curr_episode += 1

        if is_decaying_eps:
            if eps - eps_incr >= start_eps:
                eps -= eps_incr

    return q, stats


def eval_episode(env: FourRooms, q, start_state, max_steps=100):
    curr_state = env.reset(state=start_state)
    is_done = False
    curr_step = 0
    while not is_done and curr_step < max_steps:
        action = greedy(q, curr_state, env.n_actions)
        next_state, reward, is_done, info = env.step(action)
        curr_state = next_state
        curr_step += 1

    return env.curr_return, env.episode_step


def main():
    env = FourRooms()
    q, stats = golearning(env, 1000)
    print(stats)

    eps_return, eps_step = eval_episode(env, q, (1, 1))
    print(f"[EVAL] start state = (1,1), return = {eps_return}, steps = {eps_step}")
    eps_return, eps_step = eval_episode(env, q, (11, 11))
    print(f"[EVAL] start state = (11,11), return = {eps_return}, steps = {eps_step}")


if __name__ == "__main__":
    main()
