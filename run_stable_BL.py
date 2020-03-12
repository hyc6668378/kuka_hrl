from stable_baselines import PPO2
from env.KukaGymEnv import KukaDiverseObjectEnv
from stable_baselines.common.vec_env import SubprocVecEnv

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

env_fn = lambda : KukaDiverseObjectEnv(renders=False,
                                       maxSteps=32,
                                       blockRandom=0.2,
                                       actionRepeat=200, numObjects=1,
                                       single_img=False,
                                       isTest=False,
                                       verbose=True)
if __name__ == '__main__':
    env = SubprocVecEnv( [env_fn]*4 )

    model = PPO2.load('model/ppo_2_img/2', env=env, seed=0, verbose=2, gamma=0.9,
                      learning_rate=2.5e-4, nminibatches=4, n_steps=128,
                      cliprange_vf=-1, max_grad_norm=0.5,  # 不可少
                      tensorboard_log='logs/ppo2')
    model.save('model/ppo_2_img/init')
    _ = os.system("clear")

    for i in range(20):
        _ = model.learn(total_timesteps=int(5e+4), reset_num_timesteps=False)
        model.save('model/ppo_2_img/'+str(i))