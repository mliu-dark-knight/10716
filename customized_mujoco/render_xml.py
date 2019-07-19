
def render_cheetah(size):
    xml = """ <mujoco model="cheetah">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true" settotalmass="14"/>
  <default>
    <joint armature=".1" damping=".01" limited="true" solimplimit="0 .8 .03" solreflimit=".02 1" stiffness="8"/>
    <geom conaffinity="0" condim="3" contype="1" friction=".4 .1 .1" rgba="0.8 0.6 .4 1" solimp="0.0 0.8 0.01" solref="0.02 1"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <size nstack="300000" nuser_geom="1"/>
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <body name="torso" pos="0 0 .7">
      <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
      <joint armature="0" axis="1 0 0" damping="0" limited="false" name="rootx" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 0 1" damping="0" limited="false" name="rootz" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="false" name="rooty" pos="0 0 0" stiffness="0" type="hinge"/>
      <geom fromto="-.5 0 0 .5 0 0" name="torso" size="0.046" type="capsule"/>
      <geom axisangle="0 1 0 .87" name="head" pos=".6 0 .1" size="0.046 .15" type="capsule"/>
      <!-- <site name='tip'  pos='.15 0 .11'/>-->"""
    xml += "\n"
    xml +="""<body name="bthigh" pos="-.5 0 0">
        <joint axis="0 1 0" damping="6" name="bthigh" pos="0 0 0" range="-.52 1.05" stiffness="240" type="hinge"/>
        <geom axisangle="0 1 0 -3.8" name="bthigh" pos=".1 0 -.13" size="{} .145" type="capsule"/>
        <body name="bshin" pos=".16 0 -.25">
          <joint axis="0 1 0" damping="4.5" name="bshin" pos="0 0 0" range="-.785 .785" stiffness="180" type="hinge"/>
          <geom axisangle="0 1 0 -2.03" name="bshin" pos="-.14 0 -.07" rgba="0.9 0.6 0.6 1" size="{} .15" type="capsule"/>
          <body name="bfoot" pos="-.28 0 -.14">
            <joint axis="0 1 0" damping="3" name="bfoot" pos="0 0 0" range="-.4 .785" stiffness="120" type="hinge"/>
            <geom axisangle="0 1 0 -.27" name="bfoot" pos=".03 0 -.097" rgba="0.9 0.6 0.6 1" size="{} .094" type="capsule"/>
          </body>
        </body>
      </body>""".format(size[0], size[1], size[2])
    xml += "\n"
    xml +="""<body name="fthigh" pos=".5 0 0">
        <joint axis="0 1 0" damping="4.5" name="fthigh" pos="0 0 0" range="-1 .7" stiffness="180" type="hinge"/>
        <geom axisangle="0 1 0 .52" name="fthigh" pos="-.07 0 -.12" size="{} .133" type="capsule"/>
        <body name="fshin" pos="-.14 0 -.24">
          <joint axis="0 1 0" damping="3" name="fshin" pos="0 0 0" range="-1.2 .87" stiffness="120" type="hinge"/>
          <geom axisangle="0 1 0 -.6" name="fshin" pos=".065 0 -.09" rgba="0.9 0.6 0.6 1" size="{} .106" type="capsule"/>
          <body name="ffoot" pos=".13 0 -.18">
            <joint axis="0 1 0" damping="1.5" name="ffoot" pos="0 0 0" range="-.5 .5" stiffness="60" type="hinge"/>
            <geom axisangle="0 1 0 -.6" name="ffoot" pos=".045 0 -.07" rgba="0.9 0.6 0.6 1" size="{} .07" type="capsule"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>""".format(size[3], size[4], size[5])
    xml += "\n"
    xml += """<actuator>
    <motor gear="120" joint="bthigh" name="bthigh"/>
    <motor gear="90" joint="bshin" name="bshin"/>
    <motor gear="60" joint="bfoot" name="bfoot"/>
    <motor gear="120" joint="fthigh" name="fthigh"/>
    <motor gear="60" joint="fshin" name="fshin"/>
    <motor gear="30" joint="ffoot" name="ffoot"/>
  </actuator>
</mujoco> """
    return xml


def render_ant(size):
    xml = """<mujoco model="ant">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>
  <custom>
    <numeric data="0.0 0.0 0.55 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0" name="init_qpos"/>
  </custom>
  <default>
    <joint armature="1" damping="{}" limited="true"/>
    <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
  </default>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <body name="torso" pos="0 0 0.75">
      <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
      <geom name="torso_geom" pos="0 0 0" size="0.25" type="sphere"/>
      <joint armature="0" damping="0" limited="false" margin="0.01" name="root" pos="0 0 0" type="free"/>
      <body name="front_left_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 0.2 0.2 0.0" name="aux_1_geom" size="0.08" type="capsule"/>
        <body name="aux_1" pos="0.2 0.2 0">
          <joint axis="0 0 1" name="hip_1" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 0.2 0.2 0.0" name="left_leg_geom" size="0.08" type="capsule"/>
          <body pos="0.2 0.2 0">
            <joint axis="-1 1 0" name="ankle_1" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 0.4 0.4 0.0" name="left_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="front_right_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 -0.2 0.2 0.0" name="aux_2_geom" size="0.08" type="capsule"/>
        <body name="aux_2" pos="-0.2 0.2 0">
          <joint axis="0 0 1" name="hip_2" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 -0.2 0.2 0.0" name="right_leg_geom" size="0.08" type="capsule"/>
          <body pos="-0.2 0.2 0">
            <joint axis="1 1 0" name="ankle_2" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -0.4 0.4 0.0" name="right_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="back_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 -0.2 -0.2 0.0" name="aux_3_geom" size="0.08" type="capsule"/>
        <body name="aux_3" pos="-0.2 -0.2 0">
          <joint axis="0 0 1" name="hip_3" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 -0.2 -0.2 0.0" name="back_leg_geom" size="0.08" type="capsule"/>
          <body pos="-0.2 -0.2 0">
            <joint axis="-1 1 0" name="ankle_3" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -0.4 -0.4 0.0" name="third_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="right_back_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 0.2 -0.2 0.0" name="aux_4_geom" size="0.08" type="capsule"/>
        <body name="aux_4" pos="0.2 -0.2 0">
          <joint axis="0 0 1" name="hip_4" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 0.2 -0.2 0.0" name="rightback_leg_geom" size="0.08" type="capsule"/>
          <body pos="0.2 -0.2 0">
            <joint axis="1 1 0" name="ankle_4" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 0.4 -0.4 0.0" name="fourth_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_4" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_4" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_1" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_1" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_2" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_2" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_3" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_3" gear="150"/>
  </actuator>
</mujoco>""".format(size)
    return xml

def render_inverted_pendulum(pole_len):
    xml = """<mujoco model="inverted pendulum">
  <compiler inertiafromgeom="true"/>
  <default>
    <joint armature="0" damping="1" limited="true"/>
    <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
    <tendon/>
    <motor ctrlrange="-3 3"/>
  </default>
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.02"/>
  <size nstack="3000"/>
  <worldbody>
    <!--geom name="ground" type="plane" pos="0 0 0" /-->
    <geom name="rail" pos="0 0 0" quat="0.707 0 0.707 0" rgba="0.3 0.3 0.7 1" size="0.02 1" type="capsule"/>
    <body name="cart" pos="0 0 0">
      <joint axis="1 0 0" limited="true" name="slider" pos="0 0 0" range="-1 1" type="slide"/>
      <geom name="cart" pos="0 0 0" quat="0.707 0 0.707 0" size="0.1 0.1" type="capsule"/>
      <body name="pole" pos="0 0 0">
        <joint axis="0 1 0" name="hinge" pos="0 0 0" range="-90 90" type="hinge"/>
        <geom fromto="0 0 0 0.001 0 0.6" name="cpole" rgba="0 0.7 0.7 1" size="0.049 {}" type="capsule"/>
        <!--                 <body name="pole2" pos="0.001 0 0.6"><joint name="hinge2" type="hinge" pos="0 0 0" axis="0 1 0"/><geom name="cpole2" type="capsule" fromto="0 0 0 0 0 0.6" size="0.05 0.3" rgba="0.7 0 0.7 1"/><site name="tip2" pos="0 0 .6"/></body>-->
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor gear="100" joint="slider" name="slide"/>
  </actuator>
</mujoco>
""".format(pole_len)
    return xml

def render_walker2d():
    walker2d = """
    <mujoco model="walker2d">
      <compiler angle="degree" coordinate="global" inertiafromgeom="true"/>
      <default>
        <joint armature="0.01" damping=".1" limited="true"/>
        <geom conaffinity="0" condim="3" contype="1" density="1000" friction=".7 .1 .1" rgba="0.8 0.6 .4 1"/>
      </default>
      <option integrator="RK4" timestep="0.002"/>
      <worldbody>
        <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
        <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane" material="MatPlane"/>
        <body name="torso" pos="0 0 1.25">
          <camera name="track" mode="trackcom" pos="0 -3 1" xyaxes="1 0 0 0 0 1"/>
          <joint armature="0" axis="1 0 0" damping="0" limited="false" name="rootx" pos="0 0 0" stiffness="0" type="slide"/>
          <joint armature="0" axis="0 0 1" damping="0" limited="false" name="rootz" pos="0 0 0" ref="1.25" stiffness="0" type="slide"/>
          <joint armature="0" axis="0 1 0" damping="0" limited="false" name="rooty" pos="0 0 1.25" stiffness="0" type="hinge"/>
          <geom friction="0.9" fromto="0 0 1.45 0 0 1.05" name="torso_geom" size="0.05" type="capsule"/>
          <body name="thigh" pos="0 0 1.05">
            <joint axis="0 -1 0" name="thigh_joint" pos="0 0 1.05" range="-150 0" type="hinge"/>
            <geom friction="0.9" fromto="0 0 1.05 0 0 0.6" name="thigh_geom" size="0.05" type="capsule"/>
            <body name="leg" pos="0 0 0.35">
              <joint axis="0 -1 0" name="leg_joint" pos="0 0 0.6" range="-150 0" type="hinge"/>
              <geom friction="0.9" fromto="0 0 0.6 0 0 0.1" name="leg_geom" size="0.04" type="capsule"/>
              <body name="foot" pos="0.2/2 0 0.1">
                <joint axis="0 -1 0" name="foot_joint" pos="0 0 0.1" range="-45 45" type="hinge"/>
                <geom friction="0.9" fromto="-0.0 0 0.1 0.2 0 0.1" name="foot_geom" size="0.06" type="capsule"/>
              </body>
            </body>
          </body>
          <!-- copied and then replace thigh->thigh_left, leg->leg_left, foot->foot_right -->
          <body name="thigh_left" pos="0 0 1.05">
            <joint axis="0 -1 0" name="thigh_left_joint" pos="0 0 1.05" range="-150 0" type="hinge"/>
            <geom friction="0.9" fromto="0 0 1.05 0 0 0.6" name="thigh_left_geom" rgba=".7 .3 .6 1" size="0.05" type="capsule"/>
            <body name="leg_left" pos="0 0 0.35">
              <joint axis="0 -1 0" name="leg_left_joint" pos="0 0 0.6" range="-150 0" type="hinge"/>
              <geom friction="0.9" fromto="0 0 0.6 0 0 0.1" name="leg_left_geom" rgba=".7 .3 .6 1" size="0.04" type="capsule"/>
              <body name="foot_left" pos="0.2/2 0 0.1">
                <joint axis="0 -1 0" name="foot_left_joint" pos="0 0 0.1" range="-45 45" type="hinge"/>
                <geom friction="1.9" fromto="-0.0 0 0.1 0.2 0 0.1" name="foot_left_geom" rgba=".7 .3 .6 1" size="0.06" type="capsule"/>
              </body>
            </body>
          </body>
        </body>
      </worldbody>
      <actuator>
        <!-- <motor joint="torso_joint" ctrlrange="-100.0 100.0" isctrllimited="true"/>-->
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100" joint="thigh_joint"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100" joint="leg_joint"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100" joint="foot_joint"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100" joint="thigh_left_joint"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100" joint="leg_left_joint"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100" joint="foot_left_joint"/>
        <!-- <motor joint="finger2_rot" ctrlrange="-20.0 20.0" isctrllimited="true"/>-->
      </actuator>
        <asset>
            <texture type="skybox" builtin="gradient" rgb1=".4 .5 .6" rgb2="0 0 0"
                width="100" height="100"/>
            <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
            <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
            <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
            <material name="geom" texture="texgeom" texuniform="true"/>
        </asset>
    </mujoco>
    """
    return walker2d

def render_hopper():
    hopper = """
    <mujoco model="hopper">
      <compiler angle="degree" coordinate="global" inertiafromgeom="true"/>
      <default>
        <joint armature="1" damping="1" limited="true"/>
        <geom conaffinity="1" condim="1" contype="1" margin="0.001" material="geom" rgba="0.8 0.6 .4 1" solimp=".8 .8 .01" solref=".02 1"/>
        <motor ctrllimited="true" ctrlrange="-.4 .4"/>
      </default>
      <option integrator="RK4" timestep="0.002"/>
      <visual>
        <map znear="0.02"/>
      </visual>
      <worldbody>
        <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
        <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="20 20 .125" type="plane" material="MatPlane"/>
        <body name="torso" pos="0 0 1.25">
          <camera name="track" mode="trackcom" pos="0 -3 1" xyaxes="1 0 0 0 0 1"/>
          <joint armature="0" axis="1 0 0" damping="0" limited="false" name="rootx" pos="0 0 0" stiffness="0" type="slide"/>
          <joint armature="0" axis="0 0 1" damping="0" limited="false" name="rootz" pos="0 0 0" ref="1.25" stiffness="0" type="slide"/>
          <joint armature="0" axis="0 1 0" damping="0" limited="false" name="rooty" pos="0 0 1.25" stiffness="0" type="hinge"/>
          <geom friction="0.9" fromto="0 0 1.45 0 0 1.05" name="torso_geom" size="0.05" type="capsule"/>
          <body name="thigh" pos="0 0 1.05">
            <joint axis="0 -1 0" name="thigh_joint" pos="0 0 1.05" range="-150 0" type="hinge"/>
            <geom friction="0.9" fromto="0 0 1.05 0 0 0.6" name="thigh_geom" size="0.05" type="capsule"/>
            <body name="leg" pos="0 0 0.35">
              <joint axis="0 -1 0" name="leg_joint" pos="0 0 0.6" range="-150 0" type="hinge"/>
              <geom friction="0.9" fromto="0 0 0.6 0 0 0.1" name="leg_geom" size="0.04" type="capsule"/>
              <body name="foot" pos="0.13/2 0 0.1">
                <joint axis="0 -1 0" name="foot_joint" pos="0 0 0.1" range="-45 45" type="hinge"/>
                <geom friction="2.0" fromto="-0.13 0 0.1 0.26 0 0.1" name="foot_geom" size="0.06" type="capsule"/>
              </body>
            </body>
          </body>
        </body>
      </worldbody>
      <actuator>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="thigh_joint"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="leg_joint"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="foot_joint"/>
      </actuator>
        <asset>
            <texture type="skybox" builtin="gradient" rgb1=".4 .5 .6" rgb2="0 0 0"
                width="100" height="100"/>
            <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
            <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
            <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
            <material name="geom" texture="texgeom" texuniform="true"/>
        </asset>
    </mujoco>
    """
    return hopper