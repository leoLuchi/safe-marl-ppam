import operator
import random

import gymnasium as gym
import numpy as np
from numpy.ma.extras import ndenumerate


class CPAgent:
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.999,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        self.q_values = {}

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.n_agents = env.action_space.sample().shape[0]
        self.n_actions = int(gym.spaces.flatdim(env.action_space) / self.n_agents)
        self.action_shape = tuple(self.n_actions for _ in range(self.n_agents))

    def get_action(self, obs):
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self._sample(obs)
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return self._get_argmax(self._getQA(obs))

    def update(
            self,
            obs,
            action,
            reward,
            done,
            next_obs
    ):
        """Updates the Q-value of an action."""
        g = reward
        action = tuple(action)

        # if not at the end of the episode
        if next_obs is not None and obs != next_obs:
            q_next_obs = np.max(self._getQA(next_obs))
            g += self.discount_factor * q_next_obs  # expected value in next state

        q_obs = self._getQA(obs)[action]
        delta = (g - q_obs)
        q = self.lr * delta
        self._addQ(obs, action, q)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def _getQA(self, obs):
        state = self._state_from_obs(obs)
        if state not in self.q_values:
            return self._get_new_state_q_values(obs)
        return self.q_values[state]

    def _addQ(self, obs, a, q):
        if q == 0:
            return
        state = self._state_from_obs(obs)
        if state not in self.q_values:
            self.q_values[state] = self._get_new_state_q_values(obs)
        self.q_values[state][a] += q

    def _get_new_state_q_values(self, obs):
        return np.zeros(self.action_shape)

    def _state_from_obs(self, obs):
        return tuple(obs['agent_pos'])

    def _get_argmax(self, q):
        action = []
        for i in range(self.n_agents):
            if i != self.n_agents - 1:
                div = self.n_actions ** (self.n_agents - i - 1)
                action.append(int(np.argmax(q) / div) % div)
            else:
                action.append(np.argmax(q) % self.n_actions)
        return action

    def _sample(self, obs):
        return self.env.action_space.sample()


class MAPPAMAgent(CPAgent):
    def __init__(self, env: gym.Env, learning_rate: float, initial_epsilon: float, epsilon_decay: float,
                 final_epsilon: float, discount_factor: float):
        super().__init__(env, learning_rate, initial_epsilon, epsilon_decay, final_epsilon, discount_factor)
        self.agnostic_masks = {}
        self.matrix_masks = {}

    FOOD = [
        "chips",
        "biscuits",
    ]

    DRINKS = [
        "coke",
        "beer",
    ]

    PEOPLE = ["john", "mary", "alice"]

    POSSIBLE_COLLISION_LOCATIONS = [
        "N",
        "E",
        "S",
        "W",
        "NW",
        "NE",
        "SE",
        "SW",
        "NN",
        "SS",
        "WW",
        "EE",
    ]

    def _sample(self, obs):
        sample = (self.env.action_space.sample(self._memo_agnostic_masks(obs)))
        while not self._memo_matrix_mask(obs)[tuple(sample)]:
            return self._sample(obs)
        return sample

    def _get_new_state_q_values(self, obs):
        return np.array(self._memo_matrix_mask(obs) - 1, dtype=np.float32)

    def _state_from_obs(self, obs):
        x = tuple(obs['agent_pos'])
        agent_fluents = tuple(obs['fluents'].values())
        return tuple([x, agent_fluents])

    def _memo_agnostic_masks(self, obs):
        fluents_state = self._state_from_obs(obs)[1]
        if fluents_state not in self.agnostic_masks:
            self.agnostic_masks[fluents_state] = self._get_agnostic_action_masks(obs)
        return self.agnostic_masks[fluents_state]

    def _memo_matrix_mask(self, obs):
        fluents_state = self._state_from_obs(obs)[1]
        if fluents_state not in self.matrix_masks:
            self.matrix_masks[fluents_state] = self._get_matrix_mask(obs)
        return self.matrix_masks[fluents_state]

    def _get_agnostic_action_masks(self, obs):
        fluents = obs['fluents']
        masks = ()  # e.g. masks =(np.ones(self.n_actions, dtype=np.int8), np.array([0,0,0,0,0,1,0], dtype=np.int8))
        for agt in range(self.n_agents):
            mask = np.ones(self.n_actions, dtype=np.int8)
            for loc in self.POSSIBLE_COLLISION_LOCATIONS[:4]:
                if fluents["{}-{}-WALL".format(agt, loc)]:
                    match loc:
                        case "N":
                            mask[2] = 0
                        case "W":
                            mask[0] = 0
                        case "E":
                            mask[1] = 0
                        case "S":
                            mask[3] = 0
            for person in self.PEOPLE:
                if fluents["{}-at_{}".format(agt, person)]:
                    has_beer = fluents["{}-has_beer".format(agt)]
                    if has_beer and person == 'john':
                        mask[5] = 0
                        continue
                    served_food = fluents["served_{}_food".format(person)]
                    has_food = any(fluents["{}-has_{}".format(agt, food)] for food in self.FOOD)
                    if served_food and has_food:
                        mask[5] = 0
                        continue
                    served_drink = fluents["served_{}_drink".format(person)]
                    has_drink = any(fluents["{}-has_{}".format(agt, drink)] for drink in self.DRINKS)
                    if served_drink and has_drink:
                        mask[5] = 0
            masks = masks + (mask,)
        return masks

    def _get_matrix_mask(self, obs):
        matrix = 1
        agnostic_masks = self._memo_agnostic_masks(obs)
        for i in reversed(range(self.n_agents)):
            matrix = np.array([matrix * agnostic_masks[i][j] for j in range(self.n_actions)], dtype=np.int8)

        for i in range(self.n_agents - 1):
            for j in range(self.n_agents):
                if i == j:
                    continue
                for possible_location in self.POSSIBLE_COLLISION_LOCATIONS:
                    if not (obs["fluents"][str(i) + "-" + possible_location + "-" + str(j)]):
                        continue
                    disallowed = []
                    match possible_location:
                        case "N":
                            disallowed = [[2, 3], [2, 4], [4, 3], [2, 5], [5, 3]]
                        case "S":
                            disallowed = [[3, 2], [4, 2], [3, 4], [5, 2], [3, 5]]
                        case "E":
                            disallowed = [[1, 0], [1, 4], [4, 0], [1, 5], [5, 0]]
                        case "W":
                            disallowed = [[0, 1], [4, 1], [0, 4], [5, 1], [0, 5]]
                        case "NE":
                            disallowed = [[2, 0], [1, 3]]
                        case "NW":
                            disallowed = [[2, 1], [0, 3]]
                        case "SE":
                            disallowed = [[3, 0], [1, 2]]
                        case "SW":
                            disallowed = [[3, 1], [0, 2]]
                        case "NN":
                            disallowed = [[2, 3]]
                        case "SS":
                            disallowed = [[3, 2]]
                        case "EE":
                            disallowed = [[1, 0]]
                        case "WW":
                            disallowed = [[0, 1]]
                    for _, action in enumerate(disallowed):
                        slices = []
                        for agt in range(self.n_agents):
                            slices.append(slice(action[0], action[0]+1) if agt == i else
                                          slice(action[1], action[1]+1) if agt == j else
                                          slice(self.n_actions))
                        matrix[tuple(slices)] = np.zeros(matrix[tuple(slices)].shape, dtype=int)

        return matrix


class MAPPAMIndependentAgent(MAPPAMAgent):
    def __init__(self, env: gym.Env, learning_rate: float, initial_epsilon: float, epsilon_decay: float,
                 final_epsilon: float, discount_factor: float):
        super().__init__(env, learning_rate, initial_epsilon, epsilon_decay, final_epsilon, discount_factor)
        self.q_values = {i: {} for i in range(self.n_agents)}
        self.agt_mask = {i: {} for i in range(self.n_agents)}
        # self.episode = {i: [] for i in range(self.n_agents)}

    def update(
            self,
            obs,
            action,
            reward,
            done,
            next_obs
    ):
        """Updates the Q-value of an action."""
        for i in range(self.n_agents):
            g = reward

            # if not at the end of the episode
            if next_obs is not None and obs != next_obs:
                q_next_obs = np.max(self._getQA_agt(next_obs, i))
                g += self.discount_factor * q_next_obs  # expected value in next state

            q_obs = self._getQA_agt(obs, i)[action[i]]
            delta = (g - q_obs)
            q = self.lr * delta
            self._addQ_agt(obs, action[i], q, i)

    def _addQ_agt(self, obs, a, q, agt):
        if q == 0:
            return
        state = self._state_from_obs_agt(obs, agt)
        if state not in self.q_values[agt]:
            self.q_values[agt][state] = self._get_new_state_q_values_agt(obs, agt)
        if self.q_values[agt][state][a] < 0:
            return
        assert self._memo_agt_mask(obs, agt)[a] == 1
        self.q_values[agt][state][a] = max(self.q_values[agt][state][a] + q, 0)


    def get_action(self, obs):
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        joint_action = []
        for agt in range(self.n_agents):
            # with probability epsilon return a random action to explore the environment
            if np.random.random() < self.epsilon:
                action = self._sample_agt(obs, agt)
            # with probability (1 - epsilon) act greedily (exploit)
            else:
                action = np.argmax(self._getQA_agt(obs, agt))
            joint_action.append(action)

        return joint_action

    def _getQA_agt(self, obs, agt):
        state = self._state_from_obs_agt(obs, agt)
        if state not in self.q_values[agt]:
            return self._get_new_state_q_values_agt(obs, agt)
        return self.q_values[agt][state]

    def _state_from_obs_agt(self, obs, agt):
        state = self._state_from_obs(obs)
        return tuple([state[0][agt], state[1]])

    def _sample_agt(self, obs, agt):
        masks = tuple([self._memo_agt_mask(obs, agt)])
        for _ in range(1, self.n_agents):
            masks = masks + (np.ones(self.n_actions, dtype=np.int8),)
        return self.env.action_space.sample(masks)[0]

    def _get_new_state_q_values_agt(self, obs, agt):
        return np.array(self._memo_agt_mask(obs, agt) - 1, dtype=np.float32)

    def _memo_agt_mask(self, obs, agt):
        fluents_state = self._state_from_obs(obs)[1]
        if fluents_state not in self.agt_mask[agt]:
            self._get_decomposed_masks(obs)
        return self.agt_mask[agt][fluents_state]

    def _get_decomposed_masks(self, obs):
        matrix_mask = self._memo_matrix_mask(obs)
        fluents_state = self._state_from_obs(obs)[1]
        joint_to_fix = []
        filter_dict = {}
        filter_dict_agt = {i: {j: 0 for j in range(self.n_actions)} for i in range(self.n_agents)}

        for indexes, value in ndenumerate(matrix_mask):
            if not value:
                joint_to_fix.append(indexes)

        # create a data structure to sort the unsafe "single(personal) actions" (couple (agent, action) => count)
        for joint in joint_to_fix:
            for i, a_i in enumerate(joint):
                key = f"{i},{a_i}"
                filter_dict[key] = filter_dict.get(key, 0) + 1
                filter_dict_agt[i][a_i] = filter_dict[key]

        # extract one safe joint action -> so, I remove a "single(personal) action" for each agent

        def greater_than_zero(key_value):
            k, v = key_value
            return v > 0

        joint_to_save = []
        while max(
                [len(dict(filter(greater_than_zero, j.items()))) for _, j in filter_dict_agt.items()]
        ) >= self.n_actions:
            for i in range(self.n_agents):
                if len(joint_to_save) < self.n_agents:
                    joint_to_save.append(min(filter_dict_agt[i], key=filter_dict_agt[i].get))
                elif joint_to_fix.count(tuple(joint_to_save)) == 0:
                    for j in range(self.n_agents):
                        if f"{j},{joint_to_save[j]}" in filter_dict:
                            del filter_dict[f"{j},{joint_to_save[j]}"]
                        del filter_dict_agt[j][joint_to_save[j]]
                    break
                else:
                    joint_to_save[i] = random.choice(list(filter_dict_agt[i].keys()))

        # sort
        filter_list = sorted(filter_dict.items(), key=operator.itemgetter(1), reverse=True)

        removed_action_by_agt = {}

        # remove "single(personal) actions" one by one, re-sorting the list
        while filter_list:
            # If I have removed all unsafe joint actions I stop the cycle
            if not joint_to_fix:
                break
            key = filter_list.pop(0)[0]
            key = key.split(',')
            i = int(key[0])
            a_i = int(key[1])
            new_err_to_fix = []
            # For each unsafe action left I check if this single(personal) action a_i is the one selected by the agent i
            for keyJ, joint in enumerate(joint_to_fix):
                # if so, I decrement by one each agent/action count from the "single(personal) actions" count list
                if a_i == joint[i]:
                    for agt, agt_act in enumerate(joint):
                        if filter_dict.get(f"{agt},{agt_act}", False):
                            filter_dict[f"{agt},{agt_act}"] = filter_dict.get(f"{agt},{agt_act}") - 1
                            if filter_dict[f"{agt},{agt_act}"] == 0:
                                del filter_dict[f"{agt},{agt_act}"]
                else:
                    new_err_to_fix.append(joint)
            # I add the "single(personal) action" to the list of unsafe actions of this agent
            removed_action_by_agt.update({i: removed_action_by_agt.get(i, []) + [a_i]})

            filter_list = sorted(filter_dict.items(), key=operator.itemgetter(1), reverse=True)
            joint_to_fix = new_err_to_fix

        for i in range(self.n_agents):
            self.agt_mask[i][fluents_state] = np.array([0 if j in removed_action_by_agt.get(i, []) else 1
                                                        for j in range(self.n_actions)], dtype=np.int8)


class IndependentQAgent(MAPPAMIndependentAgent):
    def __init__(self, env: gym.Env, learning_rate: float, initial_epsilon: float, epsilon_decay: float,
                 final_epsilon: float, discount_factor: float):
        super().__init__(env, learning_rate, initial_epsilon, epsilon_decay, final_epsilon, discount_factor)

    def _addQ_agt(self, obs, a, q, agt):
        if q == 0:
            return
        state = self._state_from_obs_agt(obs, agt)
        if state not in self.q_values[agt]:
            self.q_values[agt][state] = np.zeros(self.n_actions, dtype=np.float32)
        self.q_values[agt][state][a] += q

    def get_action(self, obs):
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        joint_action = []
        for agt in range(self.n_agents):
            # with probability epsilon return a random action to explore the environment
            if np.random.random() < self.epsilon:
                action = self.env.action_space.sample()[0]
            # with probability (1 - epsilon) act greedily (exploit)
            else:
                action = np.argmax(self._getQA_agt(obs, agt))
            joint_action.append(action)
        return joint_action

    def _getQA_agt(self, obs, agt):
        state = self._state_from_obs_agt(obs, agt)
        if state not in self.q_values[agt]:
            return np.zeros(self.n_actions, dtype=np.float32)
        return self.q_values[agt][state]

    def _state_from_obs_agt(self, obs, agt):
        x = tuple(obs['agent_pos'])
        return x[agt]
