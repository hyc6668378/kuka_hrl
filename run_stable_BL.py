from stable_baselines import PPO2
from env.KukaGymEnv import KukaDiverseObjectEnv
from stable_baselines.common.vec_env import SubprocVecEnv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

env_fn = lambda : KukaDiverseObjectEnv(renders=False,
                           maxSteps=64,
                           blockRandom=0.2,
                           actionRepeat=200, numObjects=1, rgb_only=True,
                                       single_img=True,
                           dv=1.0, isTest=False,
                                       verbose=False,
                                       phase = 1)

if __name__ == '__main__':
    env = SubprocVecEnv( [env_fn]*3 )

    # from stable_baselines.common.policies import CnnPolicy

    model = PPO2('CnnLnLstmPolicy', env, verbose=1,
                 tensorboard_log='logs/single_imgs_stable_baselines/',
                 seed=0, gamma=0.9)

    # model = PPO2.load("ppo2_phase_1", env=env, seed=1,
    #                   tensorboard_log='logs/double_imgs_stable_baselines/')

    _ = model.learn(total_timesteps=int(2e+6), reset_num_timesteps=False)
    model.save('ppo2_phase_1')