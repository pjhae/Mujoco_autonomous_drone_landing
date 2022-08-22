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

        #### Time_step
        self.time_step = 0

        #### Curriculum Factor
        # self.curriculum_fac = 0

        #### For Drone
        self.xlist_drone = []
        self.ylist_drone = []
        self.zlist_drone = []

        #### For Car
        self.xlist_car = []
        self.ylist_car = []
        self.zlist_car = []

        ### Action buffer
        self.action_buffer = np.array([0, 0, 0, 0])
        self.action_buffer_2 = np.array([0, 0, 0, 0])

        ### Input History buffer
        self.input_history_buffer = []

        ## Distance between 2 agents
        self.dist_between_agents = 0
        self.xy_Distance_between_two_agents = 0

        ## Relative desired Position vector
        self.rel_desired_heading_vec = np.array([0, 0, 0])

        ## Turn OFF Flag
        self.turn_off_flag = 0

        # Body name for collision detection
        self.car_body_array = ["Landing_box_col"]
        self.drone_body_array = ["Main_body_col_1", "Main_body_col_2", "Main_body_col_3", "Main_body_col_4"]
        self.drone_blade_array = ["FL_blade_col", "FR_blade_col", "BL_blade_col", "BR_blade_col"]

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    @property
    def is_healthy(self):
        # Initialize Healthy condition
        is_healthy = self.state_vector()[2] > -1.9 and abs(self.state_vector()[4]) < 0.8 and self.dist_between_agents < 4.0 and self.xy_Distance_between_two_agents < 3
        for i in range(self.sim.data.ncon):
            sim_contact = self.sim.data.contact[i]
            if str(self.sim.model.geom_id2name(sim_contact.geom2)) == self.car_body_array[0]:
                if str(self.sim.model.geom_id2name(sim_contact.geom1)) in self.drone_blade_array:
                    is_healthy = False
                    print("Blade Collision!! : RESET")
                    return is_healthy
        #
        # Touched_landing_box_set = set()
        # for i in range(self.sim.data.ncon):
        #     sim_contact = self.sim.data.contact[i]
        #     if str(self.sim.model.geom_id2name(sim_contact.geom2)) == self.car_body_array[0]:
        #         if str(self.sim.model.geom_id2name(sim_contact.geom1)) in self.drone_body_array:
        #             Touched_landing_box_set.add(str(self.sim.model.geom_id2name(sim_contact.geom1)))
        # # Update Turn - Off Flag
        # if len(Touched_landing_box_set) == 4:
        #     is_healthy = False
        #     self.turn_off_flag = 1
        #     print("!!!!!!!SUCCESS!!!!!!! : !!!!!!! 4 Points are Touched !!!!!!!")
        #     return is_healthy

        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy
        return done

    def step(self, action):

        #### Check turn-off Flag
        if self.turn_off_flag == 1:
            action = [0, 0, -0.1, 0]

        #### Update Action buffer
        self.action_buffer_2 = self.action_buffer
        self.action_buffer = action

        ### Update Input History buffer
        self.input_history_buffer.append(action)

        #### Do Simulation
        self.do_simulation(action, self.frame_skip)

        qpos = np.array(self.sim.data.qpos)
        qvel = np.array([action[0], action[1], action[2], 0, action[3], 0, 0, 0, 0, 0, 0.28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.set_state(qpos, qvel)

        #### Get Position INFO
        x_drone = self.state_vector()[0] - 1.5
        y_drone = self.state_vector()[1]
        z_drone = self.state_vector()[2] + 2

        x_car = self.state_vector()[10] - 0.03
        y_car = self.state_vector()[11]
        z_car = self.state_vector()[12] + 0.1

        drone_pos = np.array([x_drone, y_drone, z_drone])
        car_pos = np.array([x_car, y_car, z_car])

        # print(z_drone, z_car)

        #### Calculate Reward

        ## Distance between two agents
        Distance_between_two_agents = np.linalg.norm([drone_pos - car_pos])
        self.dist_between_agents = Distance_between_two_agents

        ## Desired heading vector
        desired_heading_vec = np.array(car_pos - drone_pos)

        ## Drone's rotation matrix
        # Roll, Pitch, Yaw
        roll_ang = self.state_vector()[3]
        pitch_ang = self.state_vector()[4]
        yaw_ang = self.state_vector()[5]

        # Rot matrix
        rot_yaw_matrix = np.array([[np.cos(yaw_ang), -np.sin(yaw_ang), 0],
                                   [np.sin(yaw_ang), np.cos(yaw_ang), 0],
                                   [0, 0, 1]])

        rot_pitch_matrix = np.array([[np.cos(pitch_ang), 0, np.sin(pitch_ang)],
                                     [0, 1, 0],
                                     [-np.sin(pitch_ang), 0, np.cos(pitch_ang)]])

        rot_roll_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(roll_ang), -np.sin(roll_ang)],
                                    [0, np.sin(roll_ang), np.cos(roll_ang)]])

        # Desired heading vector w.r.t Drone's frame
        rel_desired_heading_vec = np.transpose(rot_yaw_matrix) @ np.transpose(rot_pitch_matrix) @ np.transpose(rot_roll_matrix) @ desired_heading_vec
        self.rel_desired_heading_vec = rel_desired_heading_vec

        ## Distance Reward
        xy_Distance_between_two_agents = np.linalg.norm([drone_pos[0:2] - car_pos[0:2]])
        self.xy_Distance_between_two_agents = xy_Distance_between_two_agents
        z_Distance_between_two_agents = np.linalg.norm([drone_pos[2] - car_pos[2]])
        dist_reward = 10 - (6 * xy_Distance_between_two_agents + 20 * abs(pitch_ang))

        ## Landing Reward
        land_reward = 0
        if xy_Distance_between_two_agents < 0.35:
            land_reward = 30 / (0.1 + z_Distance_between_two_agents)
            print("Pole!")

        ## Goal Reward
        goal_reward = 0
        Touched_landing_box_set = set()
        for i in range(self.sim.data.ncon):
            sim_contact = self.sim.data.contact[i]
            if str(self.sim.model.geom_id2name(sim_contact.geom2)) == self.car_body_array[0]:
                if str(self.sim.model.geom_id2name(sim_contact.geom1)) in self.drone_body_array:
                    Touched_landing_box_set.add(str(self.sim.model.geom_id2name(sim_contact.geom1)))
        # Update Turn - Off Flag
        if len(Touched_landing_box_set) == 4:
            goal_reward = 500000
            self.turn_off_flag = 1
            print("!!!!!!! SUCCESS !!!!!!! : !!!!!!! 4 Points are Touched !!!!!!!")

        ## Collision Reward
        col_reward = 0
        for i in range(self.sim.data.ncon):
            sim_contact = self.sim.data.contact[i]
            if str(self.sim.model.geom_id2name(sim_contact.geom2)) == self.car_body_array[0]:
                if str(self.sim.model.geom_id2name(sim_contact.geom1)) in self.drone_blade_array:
                    col_reward = -100
                    print("Blade Collision!! : REWARD = -100")

        ## Overloading reward
        over_reward = - 0.2 * np.linalg.norm(np.array(self.action_buffer_2) - np.array(self.action_buffer))

        # print(5 * Distance_between_two_agents , 2 * abs(pitch_ang) , over_reward)

        ## Reward Sum
        reward = dist_reward + land_reward + goal_reward + col_reward + over_reward
        # print(reward)
        #### Append postion of Two agents
        # For reference agent
        self.xlist_drone.append(x_drone)
        self.ylist_drone.append(y_drone)
        self.zlist_drone.append(z_drone)

        # For following agent
        self.xlist_car.append(x_car)
        self.ylist_car.append(y_car)
        self.zlist_car.append(z_car)

        # ## Plotting Trajectory
        # if self.time_step == 500 - 1 :     # max_episode_steps - 1
        #     fig = plt.figure(figsize=(8, 8))
        #     ax = fig.add_subplot(111, projection='3d')
        #
        #     fig2 = plt.figure(figsize=(8, 8))
        #     b_vx = fig2.add_subplot(221)
        #     b_vy = fig2.add_subplot(222)
        #     b_vz = fig2.add_subplot(223)
        #     b_wy = fig2.add_subplot(224)
        #
        #     # Title
        #     ax.set_title("Trajectory of Drone/Car", size=20)
        #     b_vx.set_title("Vx History", size=10)
        #     b_vy.set_title("Vy History", size=10)
        #     b_vz.set_title("Vz History", size=10)
        #     b_wy.set_title("Wy History", size=10)
        #
        #     # Label
        #     ax.set_xlabel("x", size=14)
        #     ax.set_ylabel("y", size=14)
        #     ax.set_zlabel("z", size=14)
        #
        #     # Limit
        #     ax.set_ylim(-2,2)
        #     ax.set_zlim( 0,3)
        #
        #     ax.plot(self.xlist_drone[2:], self.ylist_drone[2:], self.zlist_drone[2:], color="red" ,label='Drone')
        #     ax.plot(self.xlist_car[2:], self.ylist_car[2:], self.zlist_car[2:], color="blue", label='Car')
        #
        #     b_vx.plot(np.arange(self.time_step + 1) ,np.transpose(self.input_history_buffer)[0], color="blue", label='Vx')
        #     b_vy.plot(np.arange(self.time_step + 1), np.transpose(self.input_history_buffer)[1], color="red", label='Vy')
        #     b_vz.plot(np.arange(self.time_step + 1), np.transpose(self.input_history_buffer)[2], color="green", label='Vz')
        #     b_wy.plot(np.arange(self.time_step + 1), np.transpose(self.input_history_buffer)[3], color="yellow", label='Wy')
        #
        #     plt.legend()
        #     plt.show()

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

        return np.concatenate([self.action_buffer, [self.state_vector()[4]], self.rel_desired_heading_vec])

    def reset_model(self):
        print("START!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        ## Reset all Joint to zero position
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

        ## Update obervation
        observation = self._get_obs()

        ## Initialize timestep
        self.time_step = 0

        ## Update curriculum Factor
        # self.curriculum_fac += 1

        # Clear the batch
        self.xlist_drone = []
        self.ylist_drone = []
        self.zlist_drone = []

        self.xlist_car = []
        self.ylist_car = []
        self.zlist_car = []

        self.action_buffer = np.array([0, 0, 0, 0])
        self.action_buffer_2 = np.array([0, 0, 0, 0])

        ## Initalize turn - off Flag
        self.turn_off_flag = 0

        ## Initalize history buffer
        self.input_history_buffer = []

        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

