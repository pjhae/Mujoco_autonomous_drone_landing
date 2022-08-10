import numpy as np
import matplotlib.pyplot as plt
from gym import utils
from gym.envs.mujoco import mujoco_env
from mpl_toolkits.mplot3d import Axes3D


# if you want to receive pixel data from camera using render("rgb_array",_,_,_,_)
# you should change below line <site_packages>/gym/envs/mujoco/mujoco_env.py to:
# self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, None, -1)



DEFAULT_CAMERA_CONFIG = {
    'distance': 1.5,
}


class DroneEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, xml_file='Drone_ver_1.0/drone-v1.xml', ):
        utils.EzPickle.__init__(**locals())
        self.time_step = 0

        #### For Drone
        self.xlist_drone = []
        self.ylist_drone = []
        self.zlist_drone = []

        #### For Car
        self.xlist_car = []
        self.ylist_car = []
        self.zlist_car = []

        ### Action buffer
        action_buffer = []

        ## Distance between 2 agents
        self.dist_between_agents = 0

        ## Relative desired Position vector
        self.rel_desired_heading_vec = np.array([0, 0, 0])

        ## Turn OFF Flag
        self.turn_off_flag = 0

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)


    @property
    def is_healthy(self):
        # Initialize Healthy condition
        #is_healthy = abs(self.state_vector()[2]) < 1 and abs(self.state_vector()[4]) < 1 and self.dist_between_agents < 3
        is_healthy = True
        return is_healthy


    @property
    def done(self):
        done = not self.is_healthy
        return done

    def step(self, action):

        #### Update Action buffer
        self.action_buffer = action

        #### Check turn-off Flag
        if self.turn_off_flag == 1:
            action = [0,0,0,0]

        #### Do Simulation
        self.do_simulation(action ,self.frame_skip)

        qpos = np.array(self.sim.data.qpos)
        qvel = np.array([action[0], action[1], action[2], 0, action[3], 0 ,0,0,0,0,0.1,0,0,0,0,0,0,0,0,0,0])
        #qvel = np.array([0, 0, 0.3, action[0], action[1], action[2], 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.set_state(qpos,qvel)

        #### Get Position INFO
        x_drone = self.state_vector()[0]
        y_drone = self.state_vector()[1]
        z_drone = self.state_vector()[2] + 1

        x_car   = self.state_vector()[10] + 1
        y_car   = self.state_vector()[11]
        z_car   = self.state_vector()[12] + 0.255

        drone_pos = np.array([x_drone,y_drone,z_drone])
        car_pos = np.array([x_car,y_car,z_car])


        #### Calculate Reward

        ## Distance between two agents
        Distance_between_two_agents = np.linalg.norm([drone_pos-car_pos])
        self.dist_between_agents = Distance_between_two_agents

        ## Desired heading vector
        desired_heading_vec = np.array(car_pos-drone_pos)

        ## Drone's rotation matrix
        # Roll, Pitch, Yaw
        roll_ang = self.state_vector()[3]
        pitch_ang = self.state_vector()[4]
        yaw_ang = self.state_vector()[5]

        # Rot matrix
        rot_yaw_matrix    = np.array([[np.cos(yaw_ang), -np.sin(yaw_ang), 0],
                                      [np.sin(yaw_ang),  np.cos(yaw_ang), 0],
                                      [0              ,  0              , 1]])

        rot_pitch_matrix  = np.array([[np.cos(pitch_ang), 0, np.sin(pitch_ang)],
                                     [0                 , 1,                 0],
                                     [-np.sin(pitch_ang), 0, np.cos(pitch_ang)]])

        rot_roll_matrix   = np.array([[1,                0,                 0],
                                      [0, np.cos(roll_ang), -np.sin(roll_ang)],
                                      [0, np.sin(roll_ang),  np.cos(roll_ang)]])


        # Desired heading vector w.r.t Drone's frame
        rel_desired_heading_vec = np.transpose(rot_yaw_matrix) @ np.transpose(rot_pitch_matrix) @ np.transpose(rot_roll_matrix) @ desired_heading_vec
        self.rel_desired_heading_vec = rel_desired_heading_vec

        # Calculate Reward
        reward = 10 - (Distance_between_two_agents + 5*abs(pitch_ang))

        if Distance_between_two_agents < 0.15:
            reward = 5000
            self.turn_off_flag = 1
            print("wow!!")

        #### Append postion of Two agents
        # For reference agent
        self.xlist_drone.append(x_drone)
        self.ylist_drone.append(y_drone)
        self.zlist_drone.append(z_drone)

        # For following agent
        self.xlist_car.append(x_car)
        self.ylist_car.append(y_car)
        self.zlist_car.append(z_car)

        # #### Plotting Trajectory
        if self.time_step == 700 - 1 :     # max_episode_steps - 1
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            # Title
            ax.set_title("Trajectory of Drone/Car", size=20)
            # Label
            ax.set_xlabel("x", size=14)
            ax.set_ylabel("y", size=14)
            ax.set_zlabel("z", size=14)

            # Limit
            ax.set_ylim(-1,1)

            ax.plot(self.xlist_drone[2:], self.ylist_drone[2:], self.zlist_drone[2:], color="red" ,label='Drone')
            ax.plot(self.xlist_car[2:], self.ylist_car[2:], self.zlist_car[2:], color="blue", label='Car')
            plt.legend()
            plt.show()

        #### Update time step
        self.time_step += 1

        #### Return INFOs
        done = self.done
        observation = self._get_obs()
        info = {

            'total reward': reward
        }

        return observation, reward, done, info


    def _get_obs(self):


        return np.concatenate([self.action_buffer, self.rel_desired_heading_vec])

    def reset_model(self):

        ## Reset all Joint to zero position
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

        ## Update obervation
        observation = self._get_obs()

        ## Initialize timestep
        self.time_step = 0

        # Clear the batch
        self.xlist_drone = []
        self.ylist_drone = []
        self.zlist_drone = []

        self.xlist_car = []
        self.ylist_car = []
        self.zlist_car = []

        ## Initalize turn - off Flag
        self.turn_off_flag = 0

        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


