import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.wrappers.time_limit import TimeLimit
import os
from .render_xml import *
#import os.path

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

class CustomizedMujocoEnv(mujoco_env.MujocoEnv):
    def __init__(self, fullpath, frame_skip, rgb_rendering_tracking=True):
        # allow full_path to be an arbitrary path
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self.rgb_rendering_tracking = rgb_rendering_tracking
        self._viewers = {}
        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        self._set_action_space()
        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done
        self._set_observation_space(observation)
        self.seed()


class CustomizedHalfCheetahEnv(CustomizedMujocoEnv, utils.EzPickle):
    def __init__(self, size, file_name='customized_half_cheetah.xml'):
        if not os.path.exists('env_spec'):
            os.mkdir('env_spec')
        self.model_path = os.path.join('env_spec', file_name)
        f = open(self.model_path, 'w')
        f.write(render_cheetah(size=size))
        f.close()
        CustomizedMujocoEnv.__init__(self, self.model_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

class PartiallyObservableHalfCheetahEnv(CustomizedHalfCheetahEnv):
    def __init__(self, size, file_buffer_name='partially_observable_halfcheetah.xml'):
        super(PartiallyObservableHalfCheetahEnv, self).__init__(size, file_buffer_name)

    def _get_agent_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:]
        ])
    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_agent_obs()
    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        ob = self._get_agent_obs()
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

class PartiallyObservableHopperEnv(CustomizedMujocoEnv, utils.EzPickle):
    def __init__(self,  size, file_name='customized_hopper.xml'):
        if not os.path.exists('env_spec'):
            os.mkdir('env_spec')
        self.model_path = os.path.join('env_spec', file_name)
        f = open(self.model_path, 'w')
        f.write(render_hopper())
        f.close()
        CustomizedMujocoEnv.__init__(self, self.model_path, 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_agent_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def _get_agent_obs(self):
        return self.sim.data.qpos[1:]

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_agent_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

class PartiallyObservableWalker2dEnv(CustomizedMujocoEnv, utils.EzPickle):

    def __init__(self,  size, file_name='customized_walker2d.xml'):
        if not os.path.exists('env_spec'):
            os.mkdir('env_spec')
        self.model_path = os.path.join('env_spec', file_name)
        f = open(self.model_path, 'w')
        f.write(render_walker2d())
        f.close()
        CustomizedMujocoEnv.__init__(self, self.model_path, 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_agent_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def _get_agent_obs(self):
        return self.sim.data.qpos[1:]

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_agent_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

class CustomizedAntEnv(CustomizedMujocoEnv, utils.EzPickle):
    def __init__(self,  size, file_name='customized_ant.xml'):
        if not os.path.exists('env_spec'):
            os.mkdir('env_spec')
        self.model_path = os.path.join('env_spec', file_name)
        f = open(self.model_path, 'w')
        f.write(render_ant(size=size))
        f.close()
        CustomizedMujocoEnv.__init__(self, self.model_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

class PartiallyObservableAntEnv(CustomizedAntEnv):
    def __init__(self, file_buffer_name='partially_observable_ant.xml'):
        super(PartiallyObservableAntEnv, self).__init__(file_buffer_name)

    def _get_agent_obs(self):
        return self.sim.data.qpos.flat[2:]
    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_agent_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)
    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_agent_obs()

class CustomizedInvertedPendulumEnv(CustomizedMujocoEnv, utils.EzPickle):
    def __init__(self, pole_len,  file_name='customized_ant.xml'):
        if not os.path.exists('env_spec'):
            os.mkdir('env_spec')
        self.model_path = os.path.join('env_spec', file_name)
        f = open(self.model_path, 'w')
        f.write(render_inverted_pendulum(pole_len=pole_len))
        f.close()
        CustomizedMujocoEnv.__init__(self, self.model_path, 2)
        utils.EzPickle.__init__(self)

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        done = not notdone
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

class PartiallyObservableInvertedPendulumEnv(CustomizedInvertedPendulumEnv):
    def __init__(self, file_buffer_name='partially_observable_invertedpendulum.xml'):
        super(PartiallyObservableInvertedPendulumEnv, self).__init__(file_buffer_name)

    def _get_agent_obs(self):
        return np.concatenate([self.sim.data.qpos[:1]]).ravel()

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        done = not notdone
        ob = self._get_agent_obs()
        return ob, reward, done, {}
    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_agent_obs()

class CustomizedEnvSpec(object):

    def __init__(self, max_episode_steps=None, idx=None):      
        self.max_episode_steps = max_episode_steps
        self.id = idx
        tags = {}
        self.tags = tags

        tags['wrapper_config.TimeLimit.max_episode_steps'] = max_episode_steps

    def make(self, **kwargs):
        """Instantiates an instance of the environment with appropriate kwargs"""
        if self._entry_point is None:
            raise error.Error('Attempting to make deprecated env {}. (HINT: is there a newer registered version of this env?)'.format(self.id))
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self._entry_point):
            env = self._entry_point(**_kwargs)
        else:
            cls = load(self._entry_point)
            env = cls(**_kwargs)

        # Make the enviroment aware of which spec it came from.
        env.unwrapped.spec = self

        return env

class PartiallyObservableEnv(object):
    def __init__(self, env_name, file_buffer_name):
        self.env_name = env_name
        self.file_buffer_name = file_buffer_name
    def make(self):
        if self.env_name == "PartiallyObservableInvertedPendulum":
            env = PartiallyObservableInvertedPendulumEnv(self.file_buffer_name)
        elif self.env_name == "PartiallyObservableHalfCheetah":
            env = PartiallyObservableHalfCheetahEnv([0.046]*6, self.file_buffer_name)
        elif self.env_name == "PartiallyObservableAnt":
            env = PartiallyObservableAntEnv(self.file_buffer_name)
        elif self.env_name == "PartiallyObservableHopper":
            env = PartiallyObservableHopperEnv(self.file_buffer_name)
        elif self.env_name == "PartiallyObservableWalker2d":
            env = PartiallyObservableWalker2dEnv(self.file_buffer_name)
        else:
            raise NotImplementedError
        env.unwrapped.spec = CustomizedEnvSpec(1000, self.env_name)
        env = TimeLimit(env, max_episode_steps=1000)
        return env

