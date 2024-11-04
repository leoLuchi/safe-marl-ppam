import functools
from typing import SupportsFloat

import gymnasium
from gymnasium import error, spaces

MAX_TIMESTEP = 500


# Abstraction of the agent in the environment
class Agent:
    def __init__(self, rows, cols, pos_x=0, pos_y=0):
        self.has = []
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.rows = rows
        self.cols = cols
        self.position = 0

    def pos_if_moved(self, action):
        match (action):
            case 0:
                return self._get_position(max(self.pos_x - 1, 0), self.pos_y)
            case 1:
                return self._get_position(
                    min(self.pos_x + 1, self.rows - 1), self.pos_y
                )
            case 2:
                return self._get_position(self.pos_x, max(self.pos_y - 1, 0))
            case 3:
                return self._get_position(
                    self.pos_x, min(self.pos_y + 1, self.cols - 1)
                )
            case _:
                return self.get_position()

    def move(self, action):
        if action == 0 and self.pos_x > 0:
            self.pos_x -= 1
        elif action == 1 and self.pos_x < self.rows - 1:
            self.pos_x += 1
        elif action == 2 and self.pos_y > 0:
            self.pos_y -= 1
        elif action == 3 and self.pos_y < self.cols - 1:
            self.pos_y += 1

    def _get_position(self, pos_x, pos_y):
        return pos_x + self.cols * pos_y

    def get_position(self):
        return self._get_position(self.pos_x, self.pos_y)

    def set_position(self, x, y):
        self.pos_x = x
        self.pos_y = y


class MultipleReward(SupportsFloat):
    def __init__(self, rewards):
        self.rewards = rewards

    def __float__(self):
        return sum(self.rewards)


class CPEnvMulti(gymnasium.Env):
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
        ("alice", blue, 2, 5),
    ]

    TASKS = {
        "serve_drink_john": [{"get_coke": False, "deliver_john": False}],
        "serve_drink_mary": [{"get_beer": False, "deliver_mary": False}],
        "serve_food_john": [{"get_biscuits": False, "deliver_john": False}],
        "serve_food_mary": [{"get_chips": False, "deliver_mary": False}],
        "serve_food_alice": [
            {"get_biscuits": False, "deliver_alice": False},
        ],
        "serve_drink_alice": [
            {"get_beer": False, "deliver_alice": False},
        ],
    }

    COMMON_FLUENTS = {
        "served_john_food": False,
        "served_john_drink": False,
        "served_mary_food": False,
        "served_mary_drink": False,
        "served_alice_food": False,
        "served_alice_drink": False,
    }
    AGT_FLUENTS = [
        "has_biscuits",
        "has_chips",
        "has_beer",
        "has_coke",
        "at_john",
        "at_mary",
        "at_alice",
    ]

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

    REWARD_STATES = {
        "TaskProgress": 10,
        "TaskComplete": 100,
    }

    FOOD = [
        "chips",
        "biscuits",
    ]

    DRINKS = [
        "coke",
        "beer",
    ]

    DELIVERABLES = [
        "coke",
        "beer",
        "chips",
        "biscuits",
    ]

    PEOPLE = ["john", "mary", "alice"]

    def __init__(self, rows=6, cols=6, n_agents=2, n_people=3, multipleRewards=False):
        self.agents = {i: Agent(rows, cols) for i in range(n_agents)}
        self.action_space = spaces.MultiDiscrete([6 for _ in range(n_agents)])
        agent_pos = spaces.MultiDiscrete([rows * cols for _ in range(n_agents)])
        self.n_common_fluents = 2 * n_people
        self._reset_agent_fluents()
        fluents = spaces.Dict({key: spaces.Discrete(2) for key in self.fluents.keys()})
        self.observation_space = spaces.Dict(
            {
                "agent_pos": agent_pos,
                "fluents": fluents,
            }
        )

        self.people = self.PEOPLE[:n_people]
        self.rows = rows
        self.cols = cols
        self.multipleRewards = multipleRewards
        self.max_items_held = 1
        self.num_executed_actions = 0  # number of actions in this episode
        self.map_action_fns = {4: self._do_get, 5: self._do_deliver}
        self.info = []
        self.object_per_person = {i: [] for i in self.people}
        self.locations = self.LOCATIONS
        self.tasks = {}
        self.num_collision = 0

        self.reset()

    def step(self, action):
        self.num_executed_actions += 1
        rewards = []
        if not self.action_space.contains(action):
            raise Exception("Invalid action: {}".format(action))
        collided_agents = self._get_collided_agents(action)
        for agt, agt_action in enumerate(action):
            reward = 0
            if agt_action <= 3:
                if agt in collided_agents:
                    continue
                else:
                    self.agents[agt].move(agt_action)

            else:
                reward += self.map_action_fns[agt_action](agt)
            rewards.append(reward)

        terminated = self._is_over()
        state = self._get_game_state()
        truncated = self._is_truncated()
        multipleReward = MultipleReward(rewards)
        reward = rewards if self.multipleRewards else multipleReward.__float__()
        return state, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        num_violations = self._count_violations()
        self.num_executed_actions = 0  # number of actions in this episode
        self.num_collision = 0
        self.object_per_person = {i: [] for i in self.people}
        self._reset_tasks()
        self._reset_agent_fluents()
        for i in self.agents.keys():
            self.agents[i].set_position(i * 2, 0)

        return self._get_game_state(), {"violations": num_violations}

    def render(self, mode="human", close=False):
        if mode == "console":
            print(self._get_game_state)
        elif mode == "rgb_array":
            return self._get_game_state
            # elif mode == "human":
            #     try:
            #         import pygame
            #     except ImportError as e:
            #         raise error.DependencyNotInstalled(
            #             "{}. (HINT: install pygame using `pip install pygame`".format(e)
            #         )
            #     if close:
            #         pygame.quit()
            #     else:
            #         if self.screen is None:
            #             pygame.init()
            #             self.screen = pygame.display.set_mode(
            #                 (self.dis_width, self.dis_height)
            #             )
            #             pygame.display.set_caption("Curve")
            #         clock = pygame.time.Clock()
            #
            #         self.screen.fill(self.black)
            #         pygame.display.update()
            #         clock.tick(1)
        else:
            raise error.UnsupportedMode("Unsupported render mode: " + mode)

    def _get_agents_positions(self):
        return [agent.get_position() for agent in self.agents.values()]

    def _get_game_state(self):
        return {
            "agent_pos": self._get_agents_positions(),
            "fluents": self._evaluate_agent_fluents(),
        }

    def _is_over(self):
        for task in self.tasks.values():
            if not task:
                return False
        return True

    def _item_at(self, x, y):  # which item is in this location
        r = None
        for t in self.locations:
            if t[2] == x and t[3] == y:
                r = t[0]
                break
        return r

    def _do_get(self, agent):
        """
        Called when the agent performs the "get" action, checks that the agent can really grab something and performs
        the appropriate transition
        """
        what = self._item_at(self.agents[agent].pos_x, self.agents[agent].pos_y)
        if what is None:
            return 0
        if len(self.agents[agent].has) >= self.max_items_held:
            return 0
        if what not in self.DELIVERABLES:
            return 0
        else:
            self.agents[agent].has.append(what)
            if what in self.DRINKS:
                actual_tasks = ['serve_drink_%s']
                if what != 'beer':
                    actual_tasks.append('serve_non_alcoholic_drink_%s')
            else:
                actual_tasks = ['serve_food_%s']

            for person in self.people:
                for task in actual_tasks:
                    if not self.tasks.get(task % person , True):
                        return self.REWARD_STATES["TaskProgress"]
        return 0

    def _do_deliver(self, agent):
        """
        Called when the agent performs the "deliver" action, checks that the agent can really deliver something and
        performs the appropriate transition
        """
        reward = 0
        who = self._item_at(self.agents[agent].pos_x, self.agents[agent].pos_y)
        if who not in self.people:
            return reward
        if len(self.agents[agent].has) == 0:
            return reward
        for what in self.agents[agent].has:
            self.object_per_person[who] = self.object_per_person[who] + [what]

            if what in self.DRINKS:
                actual_task = 'serve_drink_%s' % who
                if what != 'beer' and who == 'john' :
                    actual_task = 'serve_non_alcoholic_drink_john'
            else:
                actual_task = 'serve_food_%s' % who

            if not self.tasks.get(actual_task):
                self.tasks[actual_task] = True
                reward += self.REWARD_STATES["TaskComplete"]
        self.agents[agent].has = []
        return reward

    def _reset_agent_fluents(self):
        self.fluents = self.COMMON_FLUENTS.copy()
        if len(self.fluents) > self.n_common_fluents:
            self.fluents = self.fluents[:self.n_common_fluents]
        for i in range(len(self.agents)):
            for fluent in self.AGT_FLUENTS:
                self.fluents[str(i) + "-" + fluent] = False
            for loc in self.POSSIBLE_COLLISION_LOCATIONS[:4]:
                self.fluents[str(i) + "-" + loc + "-WALL"] = False
            for j in range(len(self.agents)):
                if i != j:
                    for location in self.POSSIBLE_COLLISION_LOCATIONS:
                        self.fluents[str(i) + "-" + location + "-" + str(j)] = False

    def _has_person_food(self, who):
        for deliverable in self.object_per_person[who]:
            if deliverable in self.FOOD:
                return True
        return False

    def _has_person_drink(self, who):
        for deliverable in self.object_per_person[who]:
            if deliverable in self.DRINKS:
                return True
        return False

    def _evaluate_agent_fluents(self):
        """
        Evaluate, for the current timestep, the set of fluents that are true. These fluents are the set of subformulas
        used to determine which actions the agent can perform.

        :return: a tuple containing the truth values (1 for True, 0 for False) of the fluents.
        """
        for who in self.people:
            if self._has_person_food(who):
                self.fluents["served_%s_food" % who] = True
            if self._has_person_drink(who):
                self.fluents["served_%s_drink" % who] = True

        for i in self.agents.keys():
            what = self._item_at(self.agents[i].pos_x, self.agents[i].pos_y)
            for deliverable in self.DELIVERABLES:
                self.fluents[str(i) + "-" + "has_%s" % deliverable] = len(
                    self.agents[i].has
                ) and (deliverable == self.agents[i].has[0])
            for person in self.people:
                self.fluents[str(i) + "-" + "at_%s" % person] = what == person
            for loc in self.POSSIBLE_COLLISION_LOCATIONS[:4]:
                self.fluents[str(i) + "-" + loc + "-WALL"] = self._has_wall_at_loc(
                    self.agents[i], loc
                )
            for j in self.agents.keys():
                if i != j:
                    location = self._which_possible_collision_location(
                        self.agents[i], self.agents[j]
                    )
                    for possible_location in self.POSSIBLE_COLLISION_LOCATIONS:
                        self.fluents[
                            str(i) + "-" + possible_location + "-" + str(j)
                        ] = (possible_location == location)

        return {key: int(v) for key, v in self.fluents.items()}

    def _is_truncated(self):
        return self.num_executed_actions >= MAX_TIMESTEP

    def _reset_tasks(self):
        #     TASKS = {
        #         "serve_drink_john": [{"get_coke": False, "deliver_john": False}],
        #         "serve_drink_mary": [{"get_beer": False, "deliver_mary": False}],
        #         "serve_food_john": [{"get_biscuits": False, "deliver_john": False}],
        #         "serve_food_mary": [{"get_chips": False, "deliver_mary": False}],
        #         "serve_food_alice": [
        #             {"get_biscuits": False, "deliver_alice": False},
        #         ],
        #         "serve_drink_alice": [
        #             {"get_beer": False, "deliver_alice": False},
        #         ],
        #     }
        # for task_key in self.TASKS.keys():
        #     self.tasks[task_key] = []
        #     for tasks in self.TASKS[task_key
        #         self.tasks[task_key].append(tasks.copy())

        for person in self.people:
            if person == 'john':
                self.tasks['serve_non_alcoholic_drink_%s' % person] = False
            else:
                self.tasks['serve_drink_%s' % person] = False
            self.tasks['serve_food_%s' % person] = False



    def _get_collided_agents(self, action):
        agents_positions = self._get_agents_positions()
        new_agt_position = [
            self.agents[agt].pos_if_moved(agt_action)
            for agt, agt_action in enumerate(action)
        ]
        swaps = []
        collided_agents = set()
        for j, b in enumerate(new_agt_position):
            if new_agt_position.count(b) > 1:
                collided_agents.add(j)
            for i, a in enumerate(agents_positions):
                if a == b and i != j:
                    swaps.append((i, j))
                    if (j, i) in swaps:
                        collided_agents.add(i)
                        collided_agents.add(j)

        if len(collided_agents):
            self.num_collision += len(collided_agents)

        return collided_agents

    def _count_violations(self):
        num_violations = 0
        for person in self.people:
            food_count = 0
            drink_count = 0
            for food in self.FOOD:
                if food in self.object_per_person[person]:
                    food_count += 1
            for drink in self.DRINKS:
                if drink in self.object_per_person[person]:
                    drink_count += 1
            num_violations += max(food_count - 1, 0) + max(drink_count - 1, 0)
        if "beer" in self.object_per_person["john"]:
            num_violations += self.object_per_person["john"].count("beer")
        num_violations += self.num_collision
        return num_violations

    def _which_possible_collision_location(self, agt_1, agt_2):
        if agt_1.pos_x == agt_2.pos_x:
            match (agt_1.pos_y - agt_2.pos_y):
                case -2:
                    return "SS"
                case -1:
                    return "S"
                case 1:
                    return "N"
                case 2:
                    return "NN"

        elif agt_1.pos_x == agt_2.pos_x - 1:
            match (agt_1.pos_y - agt_2.pos_y):
                case -1:
                    return "SE"
                case 0:
                    return "E"
                case 1:
                    return "NE"
        elif agt_1.pos_x == agt_2.pos_x + 1:
            match (agt_1.pos_y - agt_2.pos_y):
                case -1:
                    return "SW"
                case 0:
                    return "W"
                case 1:
                    return "NW"
        elif agt_1.pos_y == agt_2.pos_y:
            match (agt_1.pos_x - agt_2.pos_x):
                case -2:
                    return "EE"
                case 2:
                    return "WW"
        else:
            return None

    def _has_wall_at_loc(self, agt, loc):
        match loc:
            case "N":
                return agt.pos_y == 0
            case "W":
                return agt.pos_x == 0
            case "E":
                return agt.pos_x == self.rows - 1
            case "S":
                return agt.pos_y == self.cols - 1
