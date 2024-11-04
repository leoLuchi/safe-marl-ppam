import gymnasium
import numpy as np
from gymnasium import error, spaces


class CPEnv(gymnasium.Env):
    metadata = {"render.modes": ["human", "console"]}

    black = [0, 0, 0]
    white = [255, 255, 255]
    grey = [180, 180, 180]
    dgrey = [120, 120, 120]
    orange = [180, 100, 20]
    red = [200, 0, 0]
    pink = [250, 150, 150]
    green = [0, 200, 0]
    lgreen = [60, 250, 60]
    dgreen = [0, 100, 0]
    blue = [0, 0, 250]
    lblue = [80, 200, 200]
    brown = [140, 100, 40]
    dbrown = [100, 80, 0]
    gold = [230, 215, 80]
    yellow = [210, 250, 80]

    ACTION_NAMES = ["<-", "->", "^", "v", "g", "d"]
    # 0: left, 1: right, 2: up, 3: down, 4: get, 5: deliver

    LOCATIONS = [
        ("coke", red, 1, 1),
        ("beer", gold, 2, 3),
        ("chips", yellow, 3, 1),
        ("biscuits", brown, 0, 3),
        ("john", blue, 4, 2),
        ("mary", pink, 1, 4),
    ]

    TASKS = {
        "serve_drink_john": [{"get_coke": False, "deliver_john": False}],
        "serve_drink_mary": [{"get_beer": False, "deliver_mary": False}],
        "serve_snack_john": [{"get_biscuits": False, "deliver_john": False}],
        "serve_snack_mary": [{"get_chips": False, "deliver_mary": False}],
    }

    REWARD_STATES = {
        "TaskProgress": 100,
        "TaskComplete": 1000,
    }

    DELIVERABLES = ["coke", "beer", "chips", "biscuits"]

    PEOPLE = ["john", "mary"]

    def __init__(self, rows=5, cols=5):
        self.action_space = spaces.Discrete(6)
        agent_pos = spaces.Discrete(rows * cols)
        self.fluents = {
            "served_john_food": False,
            "served_john_drink": False,
            "served_mary_food": False,
            "served_mary_drink": False,
            "has_biscuits": False,
            "has_chips": False,
            "has_beer": False,
            "has_coke": False,
            "at_john": False,
            "at_mary": False,
        }
        self.observation_space = spaces.Dict(
            {
                "agent_pos": agent_pos,
                "fluents": spaces.MultiBinary(len(self.fluents)),
            }
        )
        self.rows = rows
        self.cols = cols
        self.max_items_held = 1
        self.has = []
        self.num_executed_actions = 0  # number of actions in this episode
        self.cum_reward = 0
        self.pos_x = 0
        self.pos_y = 0
        self.map_action_fns = {4: self._do_get, 5: self._do_deliver}
        self.info = []
        self.object_per_person = {i: [] for i in self.PEOPLE}
        self.locations = self.LOCATIONS
        self.tasks = self.TASKS

        self.reset()

    def step(self, action):
        self.num_executed_actions += 1
        if not self.action_space.contains(action):
            raise Exception("Invalid action: {}".format(action))
        if action <= 3:
            self._move(action)
            reward = 0
        else:
            reward = self.map_action_fns[action]()

        terminated = self._is_over()
        state = self._get_game_state()
        return state, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        self.has = []
        self.num_executed_actions = 0  # number of actions in this episode
        self.cum_reward = 0
        self.pos_x = 0
        self.pos_y = 0
        self.object_per_person = {i: [] for i in self.PEOPLE}
        self.tasks = self.TASKS
        self.fluents = {
            "served_john_food": False,
            "served_john_drink": False,
            "served_mary_food": False,
            "served_mary_drink": False,
            "has_biscuits": False,
            "has_chips": False,
            "has_beer": False,
            "has_coke": False,
            "at_john": False,
            "at_mary": False,
        }

        return self._get_game_state()

    def render(self, mode="human", close=False):
        if mode == "console":
            print(self._get_game_state)
        else:
            raise error.UnsupportedMode("Unsupported render mode: " + mode)

    def _get_game_state(self):
        agent_position = self.pos_x + self.cols * self.pos_y
        return {
            "agent_pos": agent_position,
            "fluents": self._evaluate_agent_fluents(),
        }

    def _is_over(self):
        for task_list in self.tasks.values():
            for task in task_list:
                for task_act in task.values():
                    if not task_act:
                        return False
        return True

    def _move(self, action):
        if action == 0 and self.pos_x > 0:
            self.pos_x -= 1
        elif action == 1 and self.pos_x < self.rows - 1:
            self.pos_x += 1
        elif action == 2 and self.pos_y > 0:
            self.pos_y -= 1
        elif action == 3 and self.pos_y < self.cols - 1:
            self.pos_y += 1

    def _item_at(self, x, y):  # which item is in this location
        r = None
        for t in self.locations:
            if t[2] == x and t[3] == y:
                r = t[0]
                break
        return r

    def _do_get(self):
        """
        Called when the agent performs the "get" action, checks that the agent can really grab something and performs
        the appropriate transition
        """
        reward = 0
        what = self._item_at(self.pos_x, self.pos_y)
        if what is None:
            return reward
        if len(self.has) >= self.max_items_held:
            return reward
        if what not in self.DELIVERABLES:
            return reward
        else:
            self.has.append(what)
            for task_list in self.tasks.values():
                for task in task_list:
                    if not task.get("get_%s" % what, True):
                        task["get_%s" % what] = True
                        reward += self.REWARD_STATES["TaskProgress"]

            return reward

    def _do_deliver(self):
        """
        Called when the agent performs the "deliver" action, checks that the agent can really deliver something and
        performs the appropriate transition
        """
        reward = 0
        who = self._item_at(self.pos_x, self.pos_y)
        if who not in self.PEOPLE:
            return reward
        if len(self.has) == 0:
            return reward
        for what in self.has:
            self.object_per_person[who] = self.object_per_person[who] + [what]
            for task_list in self.tasks.values():
                for task in task_list:
                    if not task.get("deliver_%s" % what, True):
                        task["deliver_%s" % what] = True
                        reward += self.REWARD_STATES["TaskComplete"]
        self.has = []
        return reward

    def _evaluate_agent_fluents(self):
        """
        Evaluate, for the current timestep, the set of fluents that are true. These fluents are the set of subformulas
        used to determine which actions the agent can perform.

        :return: a tuple containing the truth values (1 for True, 0 for False) of the fluents.
        """
        for what in self.DELIVERABLES:
            self.fluents["has_%s" % what] = False
        for who in self.PEOPLE:
            self.fluents["at_%s" % who] = False

        for who in self.PEOPLE:
            if (
                "chips" in self.object_per_person[who]
                or "biscuits" in self.object_per_person[who]
            ):
                self.fluents["served_%s_food" % who] = True
            if (
                "beer" in self.object_per_person[who]
                or "coke" in self.object_per_person[who]
            ):
                self.fluents["served_%s_drink" % who] = True
        what = self._item_at(self.pos_x, self.pos_y)

        if what in self.PEOPLE:
            self.fluents["at_%s" % what] = True

        if self.has:
            self.fluents["has_%s" % self.has[0]] = True

        return tuple(1 if v else 0 for v in self.fluents.values())
