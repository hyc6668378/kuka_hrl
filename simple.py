import numpy as np
import gym


class simple_env(gym.Env):
    def __init__(self, state_num = 6, action_num = 5, sprase_reward=False, more_diff=False):
        self.state_num = state_num
        self.action_num = action_num
        self.sprase_reward = sprase_reward
        self.more_diff = more_diff
        self.action_space = gym.spaces.Discrete(self.action_num)
        self.observation_space = gym.spaces.Discrete(self.state_num)

        assert self.action_num % 2 == 1, "action_num must be a singular num !"

        self.trans = np.zeros([self.state_num, self.action_num], dtype=np.int8)
        for i in range(self.state_num):
            # for j, a in enumerate(list(range(-(self.action_num // 2), (self.action_num // 2) + 1))):
            for j, a in enumerate(list(range(3-self.action_num, 3))):
                self.trans[i, j] = i + a

        self.trans[self.trans < 0] = 0
        self.trans[self.trans > self.state_num] = self.state_num


        self.trans[3, :3] = -1
        if self.more_diff:
            self.trans[2, -1] = -1
            self.trans[3, :-2] = -1
            self.trans[3, -1] = -1

            self.trans[4, :-2] = -1
            self.trans[4, -1] = -1
            self.trans[5, -1] = -1
        self.trans[8, 2:] = -1
        self.trans[11, 2:] = -1
        self.trans[13, 2:] = -1
        self.trans[16, :3] = -1
        self.trans[18, 1:] = -1
        self.s = 0

    def step(self, a):
        assert a in list(range(self.action_num))

        s_ = self.trans[self.s, a]

        if self.sprase_reward:
            r = -1 if s_ == -1 else s_==self.state_num
        else:
            r = -1 if s_== -1 else s_ - self.s

        done = True if s_==self.state_num or s_ == -1  else False

        self.s = s_
        info = { 'is_success': s_==self.state_num }

        return self.s, int(r), done, info

    def reset(self):
        self.s = 0
        return self.s

    def render(self, mode='human'):
        print('Current State: {}'.format(self.s))

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        return