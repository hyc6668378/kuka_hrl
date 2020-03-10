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

    model = PPO2.load('model/ppo_2_img/init', env=env, seed=0, verbose=0, gamma=0.9,
                      learning_rate=3e-4, nminibatches=16,
                      cliprange_vf=-1, max_grad_norm=None,  # 不可少
                      tensorboard_log='logs/ppo2')

    _ = os.system("clear")

    for i in range(20):
        _ = model.learn(total_timesteps=int(1e+5), reset_num_timesteps=False)
        model.save('model/ppo_2_img/'+str(i))