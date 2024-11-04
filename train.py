import gymnasium as gym
from matplotlib import pyplot as plt
from tqdm import tqdm
import cocktail_party
from agents import IndependentQAgent          # for independent Q agent
# from agents import MAPPAMAgent                for PPAM joint action learner
# from agents import CPAgent                    for joint action learner
# from agents import MAPPAMIndependentAgent     for PPAM independent Q agent

def print_info_last_n(n, infos, num_violations):
    length = 0
    total_reward = 0
    for j, info in enumerate(infos[-n:]):
        total_reward += info['episode']['r']
        length += info['episode']['l']
    print("\nReward: {}; \nLength: {};".format(float(total_reward) / n, float(length) / n))
    print("\nTotal violations: {}".format(num_violations))
    # print("\nTask: {}".format(task))


def cocktail_party_agent():
    learning_rate = 0.1
    n_episodes = 100000
    start_epsilon = 1
    epsilon_decay = start_epsilon / n_episodes   #reduce the exploration over time
    final_epsilon = 0.1
    discount_factor = 0.99  #gamma
    env = gym.make("cocktail_party-v0")

    agent = IndependentQAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount_factor
    )

    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
    infos = []
    violations = []
    n = 5000
    for i in tqdm(range(n_episodes)):
        obs, info = env.reset()
        violations.append(info['violations'])
        # tasks.append(info['tasks'])
        done = False

        if i % n == 0:
            print_info_last_n(n, infos, sum(violations))
        #     print(agent.epsilon)

        # play one episode
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # update if the environment is done
            done = terminated or truncated
            # update the agent
            agent.update(obs, action, reward, done, next_obs)
            # update the current obs
            obs = next_obs

            # update end of episode info from statistics wrapper (info['episode']) (reward lenth...)
            if info:
                infos.append(info)

        agent.decay_epsilon()

    print_info_last_n(n, infos, sum(violations))

    mean_lengths = []
    mean_rewards = []
    x = []
    length = 0
    total_reward = 0
    for j, info in enumerate(infos):
        numiter = 100
        if j and j % numiter == 0:
            mean_rewards.append(float(total_reward) / numiter)
            mean_lengths.append(float(length) / numiter)
            x.append(j / numiter)
            length = 0
            total_reward = 0

        total_reward += info['episode']['r']
        length += info['episode']['l']

        # print("reward: {0}; length: {1} \n".format(info['episode']['r'], info['episode']['l']))
    fig, axs = plt.subplots(3)
    fig.suptitle("lr= {0}, n= {1}k, eps= {2} -> {3} gamma={4}".format(
        learning_rate,
        int(n_episodes / 1000),
        start_epsilon,
        final_epsilon,
        discount_factor
    ))
    axs[0].plot(x, mean_lengths, label='Mean length')
    axs[0].plot(x, mean_rewards, label="Mean reward")
    axs[1].plot(x, mean_lengths, label='Mean length')
    axs[2].plot(x, mean_rewards, label="Mean reward")
    # plt.plot(x, mean_lengths, label='Mean length')
    # naming the x axis
    plt.xlabel('100 episodes')
    # naming the y axis
    # plt.ylabel('')
    # plt.title()
    axs[0].legend()
    plt.show()


if __name__ == "__main__":
    cocktail_party_agent()
