from stable_baselines import PPO2
from env.KukaGymEnv import KukaDiverseObjectEnv
from stable_baselines.common.vec_env import SubprocVecEnv

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

env_fn = lambda : KukaDiverseObjectEnv(renders=False,
                                       maxSteps=64,
                                       blockRandom=0.2,
                                       actionRepeat=200, numObjects=1,
                                       rgb_only=True,
                                       single_img=False,
                                       dv=1.0, isTest=False,
                                       verbose=True,
                                       phase = 1)

if __name__ == '__main__':
    env = SubprocVecEnv( [env_fn]*3 )

    model = PPO2.load("2_imgs_lstm_7", env=env, seed=0, tensorboard_log='logs/ppo2')
    for i in range(20):
        _ = model.learn(total_timesteps=int(1e+5), reset_num_timesteps=False)

        model.save('model/ppo_2_img/'+str(i))