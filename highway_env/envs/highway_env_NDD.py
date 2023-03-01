from __future__ import division, print_function, absolute_import
import numpy as np
from highway_env.envs import HighwayExitEnv
from highway_env.vehicle.control import MDPVehicle
from highway_env.vehicle.behavior import NDDVehicle
import global_val
import bisect
from collections import namedtuple
from highway_env import utils

Action = namedtuple("Action", ["cav_action", "bv_action"])
Observation = namedtuple("Observation", ["cav_observation", "bv_observation"])
Reward = namedtuple("Reward", ["cav_reward", "bv_reward"])
Indicator = namedtuple("Indicator", ["cav_indicator", "bv_indicator"])


class HighwayEnvNDD(HighwayExitEnv):
    """
    The environment for the NADE and NDE. Developed based on the highway environment given by the highway-env repo.

    **Main attributes**
        - ``.CAV_model`` AV2, the RL AV model.
        - ``.policy_frequency`` unit:hz. the frequency of decision, e.g., 1hz (vehicle will change their action every 1 second).
        - ``.min_distance`` unit:m, start of the highway.
        - ``.max_distance`` unit:m, end of the highway.
        - ``.min_velocity`` unit:m/s, minimum velocity of the vehicle.
        - ``.max_velocity`` unit:m/s, maximum velocity of the vehicle.
        - ``.min_lane`` integer, minimum lane index.
        - ``.max_lane`` integer, maximum lane index.
        - ``.delete_BV_position`` position where to delete the BV.
        - ``.cav_observation_range`` AV observation range.
        - ``.bv_observation_range`` BV observation range.
        - ``.candidate_controlled_bv_num`` number of vehicles as the candidate to control.
        - ``.cav_observation_num`` maximum number of vehicles AV can observe.
        - ``.bv_observation_num`` maximum number of vehicles BV can observe.
        - ``.generate_vehicle_mode`` simulation environment initialization method.
        - ``.cav_obs_vehs_list`` vehicles observed by AV.

    """


    DEFAULT_CONFIG = {
        "observation": {
            "type": "Kinematics"
        },
        "policy_frequency": 1,  # [Hz]
        "initial_spacing": 2,
        "screen_width": 600,
        "screen_height": 150,
        "centering_position": [0.3, 0.5],
        "lanes_count": 3,
        "minimum_distance": 15,  # 25
        "duration": 30,
        "candidate_controlled_bv_num": 1
    }

    def __init__(self, config):
        super(HighwayEnvNDD, self).__init__()
        if not config:
            raise NameError("No config input!")
        self.CAV_model = config["CAV_model"]
        self.policy_frequency = config["policy_frequency"]
        self.min_distance = config["min_distance"]
        self.max_distance = config["max_distance"]
        self.min_velocity = config["min_velocity"]
        self.max_velocity = config["max_velocity"]
        self.min_lane = config["min_lane"]
        self.max_lane = config["max_lane"]
        self.delete_BV_position = config["delete_BV_position"]
        self.cav_observation_range = config["cav_observation_range"]
        self.bv_observation_range = config["bv_observation_range"]
        self.candidate_controlled_bv_num = config["candidate_controlled_bv_num"]
        self.cav_observation_num = config["cav_observation_num"]
        self.bv_observation_num = config["bv_observation_num"]
        self.generate_vehicle_mode = config["generate_vehicle_mode"]
        self.cav_obs_vehs_list = []

        if self.generate_vehicle_mode == "NDD" or self.generate_vehicle_mode == "given_ini":
            self.CF_percent = global_val.CF_percent
            self.ff_dis = global_val.ff_dis  # dis for ff
            self.gen_length = global_val.gen_length  # stop generate BV after this length
            self.presum_list_forward = global_val.presum_list_forward  # [] CDF for car following forward
            self.speed_CDF = global_val.speed_CDF  # for both CF and FF
            self.num_r, self.num_rr, self.num_v, self.num_acc = global_val.CF_pdf_array.shape
        else:
            raise ValueError("Generate vehicle mode not supported!")

    def reset(self, given_ini=None):
        """
        Reset the environment.

        Args:
            given_ini: the given initialization if desired.

        Returns:
            initial observations, simulation initial states.
        """
        self._make_road()
        ini_data = self._make_vehicles(given_ini=given_ini)
        self.cav_obs_vehs_list = []
        super(HighwayEnvNDD, self).reset()
        self.determine_candidate_controlled_bv()
        return self.observe_cav_bv(), ini_data

    def _make_vehicles(self, given_ini=None,
                       auto_vehicle=(0, global_val.initial_CAV_position, global_val.initial_CAV_speed)):
        """
        Initialize the simulation environment. Include ego-vehicle and background vehicles.

        Args:
            given_ini: [[Lane 0], [Lane 1], [Lane 2]], Each item [Lane i] = [[x,velocity],..., [x,velocity]].
            auto_vehicle: ego-vehicle (AV) state. [lane id, longitudinal initial position, initial speed].

        Returns:
            Initial states.
        """
        if self.CAV_model == "AV2":
            ego_vehicle = MDPVehicle(self.road, self.road.network.get_lane(("a", "b", auto_vehicle[0])).position(auto_vehicle[1], 0),
                                     velocity=auto_vehicle[2])
        else:
            raise NotImplementedError('{0} does not supported..., set to AV2.'.format(self.CAV_model))

        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        other_vehicles_type = NDDVehicle
        ini_data = None
        if given_ini:
            assert self.generate_vehicle_mode == "Given_ini"

        if self.generate_vehicle_mode == "NDD":
            vehicle_list = []  # each list in this container is for vehicles in each lane (without CAV)
            lane_num = len(self.vehicle.road.network.graph["a"]["b"])
            for lane_idx in range(lane_num):
                generate_forward = True
                generate_finish = False
                vehicle_forward_list_one_lane, vehicle_backward_list_one_lane = [], []
                if lane_idx == 0:
                    back_vehicle_speed, front_vehicle_speed = auto_vehicle[2], auto_vehicle[2]
                    back_vehicle_position, front_vehicle_position = auto_vehicle[1], auto_vehicle[1]
                else:
                    rand_speed, rand_position = self._gen_NDD_veh()
                    back_vehicle_speed, front_vehicle_speed = rand_speed, rand_speed
                    back_vehicle_position, front_vehicle_position = rand_position, rand_position
                    vehicle_forward_list_one_lane.append((rand_position, rand_speed))
                    v = other_vehicles_type(self.road, self.road.network.get_lane(('a', 'b', lane_idx)).position(rand_position, 0), 0, rand_speed)
                    self.road.vehicles.append(v)

                while generate_finish is False:
                    if generate_forward is True:
                        # print(back_vehicle_speed)
                        if back_vehicle_speed < global_val.v_low:
                            presum_list = self.presum_list_forward[global_val.v_to_idx_dic[global_val.v_low]]
                        elif back_vehicle_speed > global_val.v_high:
                            presum_list = self.presum_list_forward[global_val.v_to_idx_dic[global_val.v_high]]
                        else:
                            presum_list = self.presum_list_forward[global_val.v_to_idx_dic[int(back_vehicle_speed)]]

                        # decide CF or FF
                        random_number_CF = np.random.uniform()
                        if random_number_CF > self.CF_percent:  # FF
                            rand_speed, rand_position = self._gen_NDD_veh()
                            v_generate = rand_speed
                            pos_generate = back_vehicle_position + self.ff_dis + rand_position + global_val.LENGTH

                        else:  # CF
                            random_number = np.random.uniform()
                            r_idx, rr_idx = divmod(bisect.bisect_left(presum_list, random_number), self.num_rr)
                            try:
                                r, rr = global_val.r_to_idx_dic.inverse[r_idx], global_val.rr_to_idx_dic.inverse[rr_idx]
                            except:  # If there is no NDD in this situation, initialize the downstream vehicle by the following pre-determined parameters.
                                if back_vehicle_speed > 35:
                                    r, rr = 50, -2
                                else:
                                    r, rr = 50, 2

                            # Ensure the velocity is in the boundary
                            v_generate = np.clip(back_vehicle_speed + rr, global_val.v_low, global_val.v_high)
                            pos_generate = back_vehicle_position + r + global_val.LENGTH

                        vehicle_forward_list_one_lane.append((pos_generate, v_generate))
                        back_vehicle_speed = v_generate
                        back_vehicle_position = pos_generate

                        v = other_vehicles_type(self.road, self.road.network.get_lane(('a', 'b', lane_idx)).position(pos_generate, 0), 0, v_generate)
                        self.road.vehicles.append(v)

                        if back_vehicle_position >= self.gen_length:
                            generate_forward = False
                            generate_finish = True
                vehicle_list_each_lane = vehicle_backward_list_one_lane + vehicle_forward_list_one_lane
                vehicle_list.append(vehicle_list_each_lane)

        if self.generate_vehicle_mode == "Given_ini":
            for lane_idx in range(self.max_lane + 1):
                ini_one_lane = given_ini[lane_idx]
                for i in range(len(ini_one_lane)):
                    veh_data = ini_one_lane[i]
                    x, velocity = veh_data[0], veh_data[1]
                    v = other_vehicles_type(self.road, self.road.network.get_lane(("a", "b", lane_idx)).position(x, 0), 0, velocity)
                    self.road.vehicles.append(v)

        return ini_data

    def step(self, action):
        """
        Perform an action and step the environment dynamics.

        The action is executed by the AV and controlled BV, and all other vehicles on the road performs their naturalistic
        behaviour for several simulation timesteps until the next decision making step.

        Args:
            action: AV and controlled BV actions (if applicable).

        Returns:
            a tuple (observation, terminal, info, weight of this decision moment).

        """
        # Use cav and bv action to simulate and get results/info
        cav_action = action.cav_action
        bv_action = action.bv_action
        if bv_action:
            assert len(bv_action) == len(self.controlled_bvs)
        weight = self._simulate(cav_action, bv_action)
        # self.determine_bv()
        done, cav_crash_flag, bv_crash_flag, finish_flag, bv_crash_index = self._is_terminal()
        info = {"cav_crash_flag": cav_crash_flag, "bv_crash_flag": bv_crash_flag, "finish_flag": finish_flag, "bv_crash_index": bv_crash_index, "cav_action": cav_action}
        # get episode results and observation and action indicator
        infos = self._get_infos(info, done)
        if done:
            if not infos["scene_type"]:
                print("!")

        self.determine_candidate_controlled_bv()
        observation_and_indicator = self.observe_cav_bv()
        for bv in self.road.vehicles[1:]:
            bv.actual_action = False
            # Reset each bv weight, criticality, decomposed_controlled_flag
            bv.weight = 1
            bv.criticality = 0
            bv.decomposed_controlled_flag = False
        return observation_and_indicator, done, infos, weight

    def _get_infos(self, info, done):
        # We have 2 cases to define an over scene
        # 1. AV-Crash : AV crash with BV.
        # 2. AV-Finish-Test : AV finish or BV crash with BV.

        infos = {}
        infos["scene_type"] = None
        cav_crash_flag = info["cav_crash_flag"]

        if cav_crash_flag:
            infos["scene_type"] = "AV-Crash"
        else:
            infos["scene_type"] = "AV-Finish-Test"
        return infos

    def observe_cav_bv(self):
        """
        Get AV and BV observation. Get AV action indicator matrix used for potential safety bound in decision.

        Returns:
            Observations and Action indicators.

        """

        cav_observation, cav_action_indicator = self.get_cav_observation()
        bv_observation, bv_action_indicator = self.get_bv_observation()
        observation = Observation(cav_observation=cav_observation, bv_observation=bv_observation)
        indicator = Indicator(cav_indicator=cav_action_indicator, bv_indicator=bv_action_indicator)
        return [observation, indicator]

    def determine_candidate_controlled_bv(self):
        """
        Find certain number of BVs that near the AV as candidate controlled BVs (candidate POV).

        Returns:
            All candidate controlled BVs will be added into env.controlled_bvs list.
            The controlled flag of each candidate controlled BV will be set as True.

        """
        self.controlled_bvs = []
        if self.candidate_controlled_bv_num > 0:
            for vehicle in self.road.vehicles[1:]:
                vehicle.controlled = False
            # If there is no eligible BVs in the observation range, then just control 1 or even 0 bvs
            near_bvs = self.road.get_BV_EU(self.vehicle, len(self.road.vehicles[1:]), self.cav_observation_range)  # The BVs near AV sorted according to euclidean distance
            for vehicle in near_bvs:
                vehicle.controlled = True
                self.controlled_bvs.append(vehicle)
                if len(self.controlled_bvs) == self.candidate_controlled_bv_num:
                    break

    def get_bv_observation(self):
        """
        Get candidate controlled BV observation.

        Returns:
            BV observation.
        """
        whole_bvs_observation = []
        for bv in self.controlled_bvs:
            bv_obs = bv._get_veh_obs()
            whole_bvs_observation.append(bv_obs)
        return whole_bvs_observation, None

    def get_cav_observation(self):
        """
        Get AV observation and action indicator.

        Returns:
            observations and action indicator.
        """
        action_indicator = self.vehicle.get_action_indicator(ndd_flag=False, safety_flag=True, CAV_flag=True)
        obs, cav_obs_vehs_list = self.observation.original_observe_acc_training(cav_obs_num=self.cav_observation_num, cav_observation_range=self.cav_observation_range)
        self.cav_obs_vehs_list = cav_obs_vehs_list
        return obs, action_indicator

    def _is_terminal(self):
        """
        The episode is terminate when:
        - a collision occurs
        - AV has finished the pre-determined testing distance.

        Returns:
            (bool, bool, bool, bool, bool)
        """
        # CAV crash flag and BV crash flag
        cav_crash_flag = self.vehicle.crashed
        bv_crash_flag = False
        bv_crash_index = []
        for vehicle in self.road.vehicles[1:]:
            if vehicle.crashed:
                bv_crash_flag = True
                bv_crash_index.append(self.road.vehicles.index(vehicle))
        # Finish flag
        finish_flag = (self.vehicle.position[0] >= self.EXIT_LENGTH)
        if cav_crash_flag and not bv_crash_flag:
            raise ValueError("Crash identification error!")
        terminal = cav_crash_flag or bv_crash_flag or finish_flag
        return terminal, cav_crash_flag, bv_crash_flag, finish_flag, bv_crash_index

    def _get_log_out_veh_info(self, veh, CAV_flag=False, crash_flag=False):
        """
        Get log out vehicle state information.

        Args:
            veh: the specific vehicle.
            CAV_flag: whether it is the AV.
            crash_flag: whether it is crashed.

        Returns:
            list: vehicle state information.
        """
        veh_id = "CAV" if CAV_flag else veh.id
        if crash_flag and veh is not self.vehicle:
            if veh.crashed:
                veh.mode = "Crash"
            veh.weight, veh.criticality, veh.decomposed_controlled_flag = 1, 0, False

        if CAV_flag:
            if veh.crashed:
                mode = "Crash"
            else:
                mode = "CAV"
        else:
            mode = veh.mode
        veh_info = [global_val.episode, self.time, veh_id, mode, veh.position[0], veh.position[1], veh.lane_index[2], veh.velocity, veh.heading, veh.weight, veh.criticality,
                    veh.decomposed_controlled_flag]
        return veh_info

    def _get_log_out_veh_action(self, veh, crash_flag=False):
        """
        Get log out vehicle action information.

        Args:
            veh: the specific vehicle.
            crash_flag: whether it is crashed.

        Returns:
            list: first element is the higher-level action (left/right/acc), second element is the longitudinal acceleration, third element is the steering angle.
        """
        action = [np.nan, np.nan, np.nan]
        if crash_flag:
            return action
        else:
            if veh.lane_index != veh.target_lane_index:
                if veh.lane_index[2] > veh.target_lane_index[2]:
                    action[0] = "Left"
                elif veh.lane_index[2] < veh.target_lane_index[2]:
                    action[0] = "Right"
            else:
                action[0] = veh.longi_acc
        action[1], action[2] = veh.action['acceleration'], veh.action['steering']
        return action


    def _check_whether_lane_conflict(self, front_v, front_x, behind_v, behind_x):
        """
        Check the current step and the following 2 seconds to see whether it will crash
        """
        r_now = front_x - behind_x - global_val.LENGTH
        rr_now = front_v - behind_v
        r_1_second = r_now + rr_now * global_val.simulation_resolution
        acc_front = acc_behind = global_val.acc_low
        front_dis = utils.cal_dis_with_start_end_speed(front_v, np.clip(front_v + acc_front, global_val.v_low, global_val.v_high), acc_front,
                                                       time_interval=global_val.simulation_resolution)
        behind_dis = utils.cal_dis_with_start_end_speed(behind_v, np.clip(behind_v + acc_behind, global_val.v_low, global_val.v_high), acc_behind,
                                                        time_interval=global_val.simulation_resolution)
        r_2_second = r_1_second + front_dis - behind_dis

        if r_now <= 0 or r_1_second <= 0 or r_2_second <= 0:
            return True
        else:
            return False

    def _Abort_patch_for_MOBIL_lane_conflict(self):
        """
        Check whether MOBIL BV has safety conflict with CAV and BVs.
        """
        # First find out all BVs that are planning to do the lane change by MOBIL
        MOBIL_plan_lane_change_bv_list = []
        lane_conflict_candidate_vehs_list = []  # this list contains all vehicles that planning to do lane change at the current step (NDD BV + MOBIL BV + CAV)
        for bv in self.road.vehicles[1:]:
            if bv.lane_index != bv.target_lane_index:
                lane_conflict_candidate_vehs_list.append(bv)
                if bv.mode == "MOBIL":
                    # If this bv is planning to do the lane change by MOBIL
                    MOBIL_plan_lane_change_bv_list.append(bv)
        # Add CAV if it is doing lane change
        if self.vehicle.lane_index != self.vehicle.target_lane_index:
            lane_conflict_candidate_vehs_list.append(self.vehicle)

        if len(MOBIL_plan_lane_change_bv_list) == 0:
            return

        dis_threshold = 50
        for bv in MOBIL_plan_lane_change_bv_list:
            bv_v, bv_x = bv.velocity, bv.position[0]
            # Loop for potential conflict vehs to check whether lane conflict
            for lane_conflict_candidate_veh in lane_conflict_candidate_vehs_list:
                # They should target the same lane
                if bv.target_lane_index == lane_conflict_candidate_veh.target_lane_index:
                    candidate_v, candidate_x = lane_conflict_candidate_veh.velocity, lane_conflict_candidate_veh.position[0]
                    # They should be close
                    if abs(candidate_x - bv_x) < dis_threshold:
                        conflict_flag = False
                        if candidate_x >= bv_x:
                            conflict_flag = self._check_whether_lane_conflict(front_v=candidate_v, front_x=candidate_x, behind_v=bv_v, behind_x=bv_x)
                        else:
                            conflict_flag = self._check_whether_lane_conflict(front_v=bv_v, front_x=bv_x, behind_v=candidate_v, behind_x=candidate_x)
                        if conflict_flag:
                            other_is_MOBIL = False
                            if lane_conflict_candidate_veh is not self.vehicle:
                                if lane_conflict_candidate_veh.mode == "MOBIL": other_is_MOBIL = True
                            # If the conflict vehicle is not controlled by the MOBIL or bv is behind, bv abort
                            if not other_is_MOBIL or (bv_x <= candidate_x):
                                # Reset the lane change decision
                                bv.target_lane_index, bv.mode, bv.LC_related = bv.lane_index, np.nan, np.nan
                                # Redo the decision (Not allowed to do the lane change)
                                bv.act(bv_action=None, essential_flag=False, enable_NDD_lane_change_flag=False)
                            # Otherwise, the other abort
                            else:
                                assert (other_is_MOBIL and bv_x > candidate_x)
                                lane_conflict_candidate_veh.target_lane_index, lane_conflict_candidate_veh.mode, lane_conflict_candidate_veh.LC_related = \
                                    lane_conflict_candidate_veh.lane_index, np.nan, np.nan
                                # Redo the decision (Not allowed to do the lane change)
                                lane_conflict_candidate_veh.act(bv_action=None, essential_flag=False, enable_NDD_lane_change_flag=False)

    def _simulate(self, cav_action=None, bv_action=None):
        """
        Perform the simulation.

        Args:
            cav_action: controlled cav action (if applicable).
            bv_action: controlled bv action (if applicable).

        Returns:
            float: weight (importance sampling likelihood) of this decision moment.
        """
        weight_one_step = 1
        for k in range(int(self.SIMULATION_FREQUENCY // self.policy_frequency)):
            # The policy frequency (how frequent vehicle change their action) is generally not equivalent with simulation frequency. E.g., simulation freq is 15hz, policy freq is 1hz.
            # When essential_flag == 0, vehicles (both AV and BVs) will update their actions.
            essential_flag = self.time % int(self.SIMULATION_FREQUENCY // self.policy_frequency)

            if ((cav_action is not None) or (bv_action is not None)) and essential_flag == 0:
                # Set the CAV and BV action
                if cav_action is not None:
                    self.vehicle.act(self.ACTIONS[cav_action], essential_flag=essential_flag)
                # If BV is controlled:
                if len(self.controlled_bvs):
                    for i in range(len(self.controlled_bvs)):
                        bv = self.controlled_bvs[i]
                        if type(bv_action[i]) == int:
                            _, ndd_possi, critical_possi = bv.act(self.BV_ACTIONS[bv_action[i]], essential_flag)
                            if critical_possi and ndd_possi:
                                weight_tmp = ndd_possi / critical_possi
                                weight_one_step *= weight_tmp
                        else:
                            # bv.controlled = False
                            _, ndd_possi, critical_possi = bv.act(None, essential_flag)

            # update the steering angle, etc. for the CAV
            self.vehicle.act(essential_flag=essential_flag)
            # Choose actions for all BVs that are not controlled
            self.road.act(essential_flag)
            # If now is the step to determine actions. Do Lane conflict check for all BVs
            if essential_flag == 0:
                self._Abort_patch_for_MOBIL_lane_conflict()

            # Move all entities in the road for dt time interval
            self.road.step(1 / self.SIMULATION_FREQUENCY)

            self.time += 1
            # Automatically render intermediate simulation steps if a viewer has been launched
            self._automatic_rendering()
            road_crash_flag = False
            for vehicle in self.road.vehicles:
                if vehicle.crashed:
                    road_crash_flag = True
                    break

            if road_crash_flag:
                break

        new_vehicles_list = [self.vehicle]
        for vehicle in self.road.vehicles[1:]:
            if not (vehicle.position[0] > self.delete_BV_position) and vehicle is not self.vehicle:
                new_vehicles_list.append(vehicle)
        self.road.vehicles = new_vehicles_list
        self.enable_auto_render = False
        return weight_one_step

    def _gen_NDD_veh(self, pos_low=global_val.random_veh_pos_buffer_start, pos_high=global_val.random_veh_pos_buffer_end):
        """
        Generate a initial NDD vehicle. Longitudinal position is random sampled and speed is based on naturalistic distribution.

        Args:
            pos_low: leftmost position.
            pos_high: rightmost position.

        Returns:
            float: initial speed.
            float: random position.


        """
        random_number = np.random.uniform()
        idx = bisect.bisect_left(self.speed_CDF, random_number)
        speed = global_val.v_to_idx_dic.inverse[idx]
        rand_position = round(np.random.uniform(pos_low, pos_high))
        # print(random_number, idx, speed)
        return speed, rand_position


