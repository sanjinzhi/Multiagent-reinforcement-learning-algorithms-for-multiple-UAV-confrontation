import gym
import numpy as np
import tensorflow as tf
import argparse
import itertools
import time
import os
import pickle
import code
import random

from dqn import DQN
from memory import Memory
from make_env import make_env
import general_utilities
import simple_tag_utilities


def play(episodes, is_render, is_testing, checkpoint_interval,
         weights_filename_prefix, csv_filename_prefix, batch_size,env):
    # init statistics. NOTE: simple tag specific!
    statistics_header = ["episode"]
    statistics_header.append("steps")
    statistics_header.extend(["reward_{}".format(i) for i in range(env.n)])
    statistics_header.extend(["loss_{}".format(i) for i in range(env.n)])
    statistics_header.extend(["eps_greedy_{}".format(i) for i in range(env.n)])
    statistics_header.extend(["collisions_{}".format(i) for i in range(env.n)])
    print("Collecting statistics {}:".format(" ".join(statistics_header)))
    statistics = general_utilities.Time_Series_Statistics_Store(
        statistics_header)

    for episode in range(args.episodes):
        states = env.reset()
        episode_losses = np.zeros(env.n)
        episode_rewards = np.zeros(env.n)
        collision_count = np.zeros(env.n)
        steps = 0

        while True:
            steps += 1
            mysteps=steps
            #steps
            terminal = (mysteps >= 50)

            # render
            if args.render:
                env.render()

            # act
            actions = []
            actions_onehot = []
            for i in range(env.n):
                action = dqns[i].choose_action(states[i])
                speed = 0.9 if env.agents[i].adversary else 1

                onehot_action = np.zeros(n_actions[i])
                onehot_action[action] = speed
                actions_onehot.append(onehot_action)
                actions.append(action)

            # step
            states_next, rewards, done, info = env.step(actions_onehot)

            # learn
            if not args.testing:
                size = memories[0].pointer
                batch = random.sample(range(size), size) if size < batch_size else random.sample(
                    range(size), batch_size)

                for i in range(env.n):
                    if done[i]:
                        rewards[i] -= 50

                    memories[i].remember(states[i], actions[i],
                                         rewards[i], states_next[i], done[i])

                    if memories[i].pointer > batch_size * 10:
                        history = dqns[i].learn(*memories[i].sample(batch))
                        episode_losses[i] += history.history["loss"][0]
                    else:
                        episode_losses[i] = -1

            states = states_next
            episode_rewards += rewards
            collision_count += np.array(
                simple_tag_utilities.count_agent_collisions(env))

            # reset states if done
            if (any(done) or terminal):
                mysteps=0
                episode_rewards = episode_rewards / steps
                episode_losses = episode_losses / steps

                statistic = [episode]
                statistic.append(steps)
                statistic.extend([episode_rewards[i] for i in range(env.n)])
                statistic.extend([episode_losses[i] for i in range(env.n)])
                statistic.extend([dqns[i].eps_greedy for i in range(env.n)])
                statistic.extend(collision_count.tolist())
                statistics.add_statistics(statistic)
                
                if episode % 25 == 0:
                    print(statistics.summarize_last())
                break

        if episode % checkpoint_interval == 0:
            statistics.dump("{}_{}.csv".format(csv_filename_prefix,
                                               episode))
            general_utilities.save_dqn_weights(dqns,
                                               "{}_{}_".format(weights_filename_prefix, episode))
            if episode >= checkpoint_interval:
                os.remove("{}_{}.csv".format(csv_filename_prefix,
                                             episode - checkpoint_interval))

    return statistics


if __name__ == '__main__':
    print ("Have enter main")
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='simple_tag_guided', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--episodes', default=60000, type=int)
    parser.add_argument('--render', default=False, action="store_true")
    parser.add_argument('--benchmark', default=False, action="store_true")
    parser.add_argument('--experiment_prefix', default=".",
                        help="directory to store all experiment data")
    parser.add_argument('--weights_filename_prefix', default='/save/tag-dqn',
                        help="where to store/load network weights")
    parser.add_argument('--csv_filename_prefix', default='/save/statistics-dqn',
                        help="where to store statistics")
    parser.add_argument('--checkpoint_frequency', default=500,
                        help="how often to checkpoint")
    parser.add_argument('--testing', default=False, action="store_true",
                        help="reduces exploration substantially")
    parser.add_argument('--random_seed', default=2, type=int)
    parser.add_argument('--memory_size', default=10000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epsilon_greedy', nargs='+', type=float,
                        help="Epsilon greedy parameter for each agent")
    args = parser.parse_args()

    general_utilities.dump_dict_as_json(vars(args),
                                        args.experiment_prefix + "/save/run_parameters.json")
    # init env
    print ("Have init env")
    env = make_env(args.env, args.benchmark)

    if args.epsilon_greedy is not None:
        if len(args.epsilon_greedy) == env.n:
            epsilon_greedy = args.epsilon_greedy
        else:
            raise ValueError("Must have enough epsilon_greedy for all agents")
    else:
        epsilon_greedy = [0.5 for i in range(env.n)]

    # set random seed
    print ("Have set seed")
    env.seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    # init DQNs
    print ("Have init DQN")
    n_actions = [env.action_space[i].n for i in range(env.n)]
    state_sizes = [env.observation_space[i].shape[0] for i in range(env.n)]
    memories = [Memory(args.memory_size) for i in range(env.n)]
    dqns = [DQN(n_actions[i], state_sizes[i], eps_greedy=epsilon_greedy[i])
            for i in range(env.n)]

    general_utilities.load_dqn_weights_if_exist(
        dqns, args.experiment_prefix + args.weights_filename_prefix)

    start_time = time.time()

    # play
    print ("Have play")
    statistics = play(args.episodes, args.render, args.testing,
                      args.checkpoint_frequency,
                      args.experiment_prefix + args.weights_filename_prefix,
                      args.experiment_prefix + args.csv_filename_prefix,
                      args.batch_size,env)

    # bookkeeping
    print("Finished {} episodes in {} seconds".format(args.episodes,
                                                      time.time() - start_time))
    general_utilities.save_dqn_weights(
        dqns, args.experiment_prefix + args.weights_filename_prefix)
    statistics.dump(args.experiment_prefix + args.csv_filename_prefix + ".csv")
