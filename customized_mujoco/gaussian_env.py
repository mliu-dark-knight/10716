class GaussianEnvPool(object):
    def __init__(self, file_buffer_name, scale=0.1):
        self.envs = []
        self.scale = scale
        self.file_buffer_name = file_buffer_name
        self.cur_env = None
    def sample_env(self):
        pass

class GaussianHalfCheetahEnv(GaussianEnvPool):
    def __init__(self, *args,**kwargs):
        super(GaussianHalfCheetahEnv, self).__init__(*args, **kwargs)

    def sample_env(self):
        size = np.random.normal(loc=[0.046]*6, scale=[self.scale]*6)
        size = np.random.normal(loc=[0.046]*6, scale=[self.scale]*6)
        size = np.minimum(size, 0.046-5*self.scale)
        size = np.maximum(size, 0.046+5*self.scale)
        env = CustomizedHalfCheetahEnv(size, self.file_buffer_name)
        env.unwrapped.spec = CustomizedEnvSpec(999)
        env = TimeLimit(env, max_episode_steps=999)
        self.cur_env = env
        return env

class GaussianAntEnv(GaussianEnvPool):
    def __init__(self, *args,**kwargs):
        super(GaussianAntEnv, self).__init__(*args, **kwargs)
        self.original_damping = 1

    def sample_env(self):
        damping = 0.4*np.random.random()-0.2+self.original_damping
        env = CustomizedAntEnv(damping, self.file_buffer_name)
        env.unwrapped.spec = CustomizedEnvSpec(999)
        env = TimeLimit(env, max_episode_steps=999)
        self.cur_env = env
        return env

class GaussianInvertedPendulumEnv(GaussianEnvPool):
    def __init__(self, *args,**kwargs):
        super(GaussianInvertedPendulumEnv, self).__init__(*args, **kwargs)
        self.original_len = 0.3

    def sample_env(self):
        pole_len = 0.5*np.random.random()-0.25+self.original_len
        env = CustomizedInvertedPendulumEnv(pole_len, self.file_buffer_name)
        env.unwrapped.spec = CustomizedEnvSpec(999)
        env = TimeLimit(env, max_episode_steps=999)
        self.cur_env = env
        return env
