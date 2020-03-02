# coding=utf-8
from alg.ddpg import DDPG
from alg.OUNoise import OUNoise
import numpy as np
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
import random
from alg.toolkit import mkdir
from mpi4py import MPI

def common_arg_parser():
    parser = argparse.ArgumentParser()

    # common argument
    parser.add_argument('--memory_size',    type=int, default=2019, help="MEMORY_CAPACITY. default = 2019")
    parser.add_argument('--inter_learn_steps', type=int, default=50, help="一个step中agent.learn()的次数. default = 50")
    parser.add_argument('--experiment_name',   type=str, default='no_name', help="实验名字")
    parser.add_argument('--batch_size',    type=int, default=16, help="batch_size. default = 16")
    parser.add_argument('--max_ep_steps',    type=int, default=20, help="一个episode最大长度. default = 50")
    parser.add_argument('--seed',    type=int, default=0, help="random seed. default = 0")
    parser.add_argument('--isRENDER',  action="store_true", help="Is render GUI in evaluation?")
    parser.add_argument('--max_epochs', type=int, default=int(1e+4), help="The max_epochs of whole training. default = 10000")
    parser.add_argument("--noise_target_action", help="noise target_action for Target policy smoothing", action="store_true")
    parser.add_argument("--nb_rollout_steps", type=int, default=100, help="The timestep of rollout. default = 100")
    parser.add_argument("--evaluation", help="Evaluate model", action="store_true")
    parser.add_argument("--LAMBDA_predict", type=float, default=0.5, help="auxiliary predict weight. default = 0.5")
    parser.add_argument("--use_segmentation_Mask", help="Evaluate model", action="store_true")

    # priority memory replay
    parser.add_argument("-p", "--priority", action="store_true", help="priority memory replay")
    parser.add_argument('--alpha', type=float, default=0.6, help="priority degree")
    parser.add_argument("--turn_beta",  action="store_true", help="turn the beta from 0.6 to 1.0")

    # n_step_return
    parser.add_argument("--use_n_step", help="use n_step_loss", action="store_true")
    parser.add_argument('--n_step_return', type=int, default=5, help="n step return. default = 5")

    # DDPGfD
    parser.add_argument("--use_DDPGfD",  action="store_true", help="train with Demonstration")
    parser.add_argument('--Demo_CAPACITY', type=int, default=0, help="The number of demo transitions. default = 2000")
    parser.add_argument('--PreTrain_STEPS', type=int, default=0, help="The steps for PreTrain. default = 2000")
    parser.add_argument("--LAMBDA_BC", type=int, default=0, help="behavior clone weight. default = 100.0")

    # TD3
    parser.add_argument("--use_TD3", help="使用TD3 避免过估计", action="store_true")
    parser.add_argument("--policy_delay", type=int, default=2, help="policy update delay w.r.t critic update. default = 2")

    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args

def set_process_seeds(myseed):

    tf.set_random_seed(myseed)
    np.random.seed(myseed)
    random.seed(myseed)

def Noise_Action(Noise, action, a_space):
    noise = Noise.sample()
    action = np.clip( action + noise,
                      a_space.low,
                      a_space.high )
    return action

def plot(succ_list, steps_list):

    plt.figure(figsize=(13, 10))
    plt.ylim((0, 100))
    plt.plot(steps_list, succ_list, label='success_rate')
    plt.title(args.experiment_name)
    plt.legend()
    plt.xlabel('train_Episode')
    plt.ylabel('success rate')
    plt.savefig('result/'+args.experiment_name+'/'+args.experiment_name+'.png')

def demo_collect(agent, Demo_CAPACITY, use_segmentation_Mask):
    print("\n Demo Collecting...")
    if use_segmentation_Mask:
        demo_dir = 'demo_with_segm/'
    else:
        demo_dir = 'all_demo/'

    with agent.sess.as_default(), agent.graph.as_default():
        for i in range(Demo_CAPACITY):
            demo_tran_file_path = demo_dir+'demo%d.npy' % (i)
            transition = np.load(demo_tran_file_path, allow_pickle=True)
            transition = transition.tolist()
            agent.store_transition(
                                   obs0=transition['obs0'],
                                   action=transition['action'],
                                   reward=transition['reward'],
                                   obs1=transition['obs1'],
                                   terminal1=transition['terminal1'],
                                   demo = True)
    print(" Demo Collection completed.")

def preTrain(agent, PreTrain_STEPS):
    print("\n PreTraining ...")
    with agent.sess.as_default(), agent.graph.as_default():
        for t_step in range(PreTrain_STEPS):
            agent.learn(t_step)

    print(" PreTraining completed.")

def train(agent, env, eval_env, max_epochs, rank, PreTrain_STEPS, turn_beta,
          nb_rollout_steps=100, inter_learn_steps=50, **kwargs):

    # OU noise
    Noise = OUNoise(size=env.action_space.shape[0], mu=0, theta=0.05, sigma=0.25)

    assert np.all(np.abs(env.action_space.low) == env.action_space.high)
    print('Process_%d start rollout!'%(rank))
    with agent.sess.as_default(), agent.graph.as_default():
        obs = env.reset()

        # book everything
        episode_length = 0
        episodes = 0
        episode_cumulate_reward_history = []
        episode_cumulate_reward = 0
        eval_episodes = 0
        train_step = PreTrain_STEPS

        for epoch in range(int(max_epochs)):
            # Perform rollouts.
            for i in range(nb_rollout_steps):

                # Predict next action.
                action = agent.pi(obs)
                action = Noise_Action(Noise, action, env.action_space)
                assert action.shape == env.action_space.shape

                new_obs, reward, done, info = env.step(action)

                agent.num_timesteps += 1

                episode_cumulate_reward += reward
                episode_length += 1

                agent.store_transition(obs, action, reward, new_obs, done, demo=False )
                obs = new_obs

                if done:
                    # Episode done.
                    episodes += 1
                    agent.save_episoed_result(episode_cumulate_reward, episode_length, info['is_success'], episodes)
                    episode_cumulate_reward_history.append(episode_cumulate_reward)
                    episode_cumulate_reward = 0
                    episode_length = 0
                    obs = env.reset()
                    Noise.reset()

            # Train.
            if agent.pointer >= 5 * agent.batch_size:
                # print('train!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                for t_train in range(inter_learn_steps):
                    agent.learn(train_step)
                    train_step += 1

            # Noise decay
            Noise.sigma = np.linspace(0.25, 0.0, max_epochs)[epoch]
            if turn_beta:
                agent.beta = np.linspace(0.6, 1.0, max_epochs)[epoch]

            # Evaluate. The frequency of evaluation is limited as below
            if eval_env is not None and epoch % 10 == 0:
                print('\neval_episodes:', eval_episodes)
                eval_episode_cumulate_reward = 0.
                eval_episode_length = 0

                eval_obs_old, eval_done = eval_env.reset(), False
                while not eval_done:
                    a = agent.pi(eval_obs_old)
                    # print("eval_action:",a)
                    eval_obs_new, r, eval_done, eval_info = env.step( a )

                    eval_episode_cumulate_reward = 0.99 * eval_episode_cumulate_reward + r
                    eval_episode_length += 1
                eval_episodes += 1
                agent.save_eval_episoed_result(eval_episode_cumulate_reward, eval_episode_length,
                                               eval_info['is_success'], eval_episodes)
                if eval_info['is_success']:
                    print('Success!')
                else:
                    print('Fail...')

        return agent

def main(experiment_name, seed, max_epochs, evaluation, isRENDER, max_ep_steps,
         use_DDPGfD, Demo_CAPACITY, PreTrain_STEPS, use_segmentation_Mask,
         **kwargs):

    # 生成实验文件夹
    rank = MPI.COMM_WORLD.Get_rank()
    mkdir( rank, experiment_name)

    # The seed is different between each process.
    seed = seed + 2019 * rank
    set_process_seeds(seed)
    print('\nrank {}: seed={}'.format(rank, seed))

    from env.KukaGymEnv import KukaDiverseObjectEnv

    env = KukaDiverseObjectEnv(renders=False,
                                          maxSteps=max_ep_steps,
                                          blockRandom=0.2,
                                          actionRepeat=200,
                                          numObjects=1, dv=1.0,
                                          isTest=False, proce_num=rank,
                                          phase = 1 )

    kwargs['obs_space'] = env.observation_space
    kwargs['action_space'] = env.action_space

    if evaluation and rank == 0:
        eval_env = KukaDiverseObjectEnv(renders=isRENDER,
                         maxSteps=max_ep_steps,
                         blockRandom=0.2,
                         actionRepeat=200,
                         numObjects=1, dv=1.0,
                         isTest=False, proce_num=rank,
                         phase=1)
    else:
        eval_env = None

    agent = DDPG(rank, **kwargs, experiment_name=experiment_name)

    if use_DDPGfD:
        demo_collect( agent, Demo_CAPACITY, use_segmentation_Mask )
        preTrain( agent, PreTrain_STEPS )

    agent_trained = train( agent, env, eval_env, max_epochs, rank, PreTrain_STEPS, **kwargs )

    if rank == 0:
        agent_trained.Save()

if __name__ == '__main__':

    args = common_arg_parser()
    main(**args)

