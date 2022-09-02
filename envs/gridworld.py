from typing import Union, Tuple, Optional, List, Set

import gym
import numpy as np
from gym.core import ActType, ObsType, RenderFrame
from gym import spaces


class FourRooms(gym.Env):
    def __init__(self,
                 random_starts: bool = True,
                 start_state: Optional[Union[int, Tuple[int, int]]] = None,
                 grid: Optional[Union[str, np.ndarray]] = None,
                 goals: Optional[Union[Set[int], Set[Tuple[int, int]]]] = None,
                 desirable_goals: Optional[Union[Set[int], Set[Tuple[int, int]]]] = None,
                 step_reward: float = -0.1,
                 desirable_reward: float = 5.0,
                 undesirable_reward: float = -5.0,
                 rmin: float = -10.0
                 ):
        if not random_starts and start_state is None:
            raise ValueError("If random_starts is False then start_state must be specified")

        self.total_steps = 0
        self.episode_step = 0
        self.curr_episode = 0  # How many resets have been done
        self.curr_return = 0

        self.start_state = start_state
        self.random_starts = random_starts

        # REWARDS #
        self.step_reward = step_reward
        self.desirable_reward = desirable_reward
        self.undesirable_reward = undesirable_reward
        self.rmin = rmin

        # GRID #
        if grid is None:
            grid = "1 1 1 1 1 1 1 1 1 1 1 1 1\n" \
                   "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
                   "1 0 0 0 0 0 0 0 0 0 0 0 1\n" \
                   "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
                   "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
                   "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
                   "1 1 0 1 1 1 1 0 0 0 0 0 1\n" \
                   "1 0 0 0 0 0 1 1 1 1 0 1 1\n" \
                   "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
                   "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
                   "1 0 0 0 0 0 0 0 0 0 0 0 1\n" \
                   "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
                   "1 1 1 1 1 1 1 1 1 1 1 1 1"

        if type(grid) == str:
            grid_lines = grid.split("\n")
            grid_arr = [[int(el) for el in line.split(" ")] for line in grid_lines]
            grid = np.asarray(grid_arr)
        elif type(grid) == np.ndarray:
            pass
        else:
            raise ValueError(f"type(grid) == {type(grid)} is not supported. grid must be of type str or np.ndarray")
        self.grid = grid

        # find valid states from grid
        valid_states_tup: List[Tuple[int, int]] = list(map(tuple, np.argwhere(self.grid == 0)))
        self.valid_states = [self._flatten_state(state_tup) for state_tup in valid_states_tup]

        if random_starts:
            self.curr_state = self._get_random_state()
        else:
            if type(start_state) == Tuple:
                start_state = self._flatten_state(start_state)
            self.curr_state = start_state

        # goals
        if goals is None:
            goals = {(3, 3), (3, 9), (9, 3), (9, 9)}

        goal_type = type(list(goals)[0])
        if goal_type == tuple:
            self._goals: set[int] = set()
            for goal in goals:
                self._goals.add(self._flatten_state(goal))
        elif goal_type == int:
            self._goals: Set[int] = goals
        else:
            raise ValueError(f"goals contains elements of type {goal_type}. "
                             f"goals must contain elements of type int or Tuple[int, int]")

        # desirable goals
        if desirable_goals is None:
            desirable_goals = {(3, 3), (3, 9)}  # Top goals

        self._desirable_goals: Set[int] = {-1}  # Temp value, will be set in set_desirable_goals
        self._undesirable_goals: Set[int] = {-1}
        self.set_desirable_goals(desirable_goals)

        # action_space and observation_space
        self.n_actions = 5
        self.n_states = self.grid.shape[0] * self.grid.shape[1]

        self.actions_map = {
            "UP": 0,
            "RIGHT": 1,
            "DOWN": 2,
            "LEFT": 3,
            "STAY": 4
        }
        self.actions_map_rev = {v: k for k, v in self.actions_map.items()}  # reverse map of actions_map

        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(self.n_actions)
        pass

    ##################
    # HELPER METHODS #
    ##################
    # Gets random valid state
    def _get_random_state(self):
        return np.random.choice(self.valid_states)

    def _flatten_state(self, state: Tuple[int, int]) -> int:
        return int(np.ravel_multi_index(state, self.grid.shape))

    def _unflatten_state(self, state: int) -> Tuple[int, int]:
        return tuple(np.unravel_index(state, self.grid.shape))

    def _in_bounds(self, state: Union[int, Tuple[int, int]]):
        if type(state) == int:
            state = self._unflatten_state(state)

        grid_shape = self.grid.shape
        return 0 <= state[0] <= grid_shape[0] and 0 <= state[1] <= grid_shape[1]

    def _state_valid(self, state: Union[int, Tuple[int, int]]) -> bool:
        if type(state) == int:
            state = self._unflatten_state(state)

        grid_shape = self.grid.shape
        in_bounds = 0 <= state[0] <= grid_shape[0] and 0 <= state[1] <= grid_shape[1]
        is_obstacle = self.grid[state] == 1
        return in_bounds and not is_obstacle

    ###################
    # GETTERS/SETTERS #
    ###################
    @property
    def goals(self):
        return self._goals

    @property
    def desirable_goals(self):
        return self._desirable_goals

    @property
    def undesirable_goals(self):
        return self._undesirable_goals

    def set_desirable_goals(self, desirable_goals: Optional[Union[Set[int], Set[Tuple[int, int]]]]):
        goal_type = type(list(desirable_goals)[0])
        if goal_type == tuple:
            self._desirable_goals = set()
            for goal in desirable_goals:
                self._desirable_goals.add(self._flatten_state(goal))
        elif goal_type == int:
            self._desirable_goals = desirable_goals
        else:
            raise ValueError(f"desirable_goals contains elements of type {goal_type}. "
                             f"desirable_goals must contain elements of type int or Tuple[int, int]")

        if not self._desirable_goals.issubset(self._goals):
            raise ValueError("desirable_goals must be a subset of goals")

        # undesirable goals
        self._undesirable_goals = self.goals.difference(self._desirable_goals)

    ###########################
    # HELPERS FOR GYM METHODS #
    ###########################
    def _take_action(self, action: Union[int, str]) -> Tuple[int, float, bool, dict]:
        if type(action) == int:
            action = self.actions_map_rev[action]

        is_done = False
        info = {"desc": ""}
        reward = self.step_reward
        curr_state = self._unflatten_state(self.curr_state)

        if action == "UP":
            cand_state = (curr_state[0] - 1, curr_state[1])
        elif action == "RIGHT":
            cand_state = (curr_state[0], curr_state[1] + 1)
        elif action == "DOWN":
            cand_state = (curr_state[0] + 1, curr_state[1])
        elif action == "LEFT":
            cand_state = (curr_state[0], curr_state[1] - 1)
        elif action == "STAY":
            cand_state = curr_state
            if self._flatten_state(curr_state) in self.desirable_goals:
                is_done = True
                reward = self.desirable_reward
                info["desc"] = "Episode terminated at desirable goal"
            elif self._flatten_state(curr_state) in self.undesirable_goals:
                is_done = True
                reward = self.undesirable_reward
                info["desc"] = "Episode terminated at undesirable goal"
        else:
            raise ValueError

        if self._state_valid(cand_state):
            next_state = self._flatten_state(cand_state)
            self.curr_state = next_state
        else:
            next_state = self.curr_state

        return next_state, reward, is_done, info

    ###############
    # GYM METHODS #
    ###############
    def step(self, action: Union[int, str]) -> Union[
        Tuple[int, float, bool, bool, dict], Tuple[int, float, bool, dict]
    ]:
        next_state, reward, is_done, info = self._take_action(action)
        self.total_steps += 1
        self.curr_return += reward
        self.episode_step += 1

        return next_state, reward, is_done, info

    # noinspection PyMethodOverriding
    def reset(self,
              state: Optional[Union[int, Tuple[int, int]]] = None,
              flatten_output: bool = True) -> Union[int, Tuple[int, int]]:
        if state is None:
            if self.random_starts:
                state = self._get_random_state()
            else:
                state = self.start_state
        if type(state) == tuple:
            state = self._flatten_state(state)

        self.curr_state = state

        self.curr_episode += 1
        self.episode_step = 0
        self.episode_step = 0
        self.curr_return = 0

        if not flatten_output:
            state = self._unflatten_state(state)

        return state

    def render(self, mode="ascii"):
        if mode == "ascii":
            grid = self.grid.tolist()
            curr_state = self._unflatten_state(self.curr_state)
            arr_map = {0: " ", 1: "#"}
            grid_str_arr = [[arr_map[el] for el in arr] for arr in grid]

            for goal in self._desirable_goals:
                goal = self._unflatten_state(goal)
                grid_str_arr[goal[0]][goal[1]] = "D"
            for goal in self._undesirable_goals:
                goal = self._unflatten_state(goal)
                grid_str_arr[goal[0]][goal[1]] = "U"

            grid_str_arr[curr_state[0]][curr_state[1]] = "X"

            grid_str = "\n".join([" ".join(line) for line in grid_str_arr])
            print(grid_str)
        else:
            raise NotImplementedError(f"mode = {mode} is not supported")
