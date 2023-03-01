from __future__ import division, print_function
import numpy as np
from highway_env.vehicle.control import ControlledVehicle
from highway_env import utils
import global_val
import bisect
import scipy.io
import scipy
import copy
import scipy.stats


class IDMVehicle(ControlledVehicle):
    """
        A vehicle using both a longitudinal and a lateral decision policies.

        - Longitudinal: the IDM model computes an acceleration given the preceding vehicle's distance and velocity.
        - Lateral: the MOBIL model decides when to change lane by maximizing the acceleration of nearby vehicles.
    """
    ACC_LIST = []
    BV_ACTIONS = {0: 'LANE_LEFT', 1: 'LANE_RIGHT'}
    num_acc = int(((global_val.acc_high - global_val.acc_low) / global_val.acc_step) + 1)
    num_non_acc = len(BV_ACTIONS)
    for i in range(num_acc):
        acc = global_val.acc_to_idx_dic.inverse[i]
        BV_ACTIONS[i + num_non_acc] = str(acc)
        ACC_LIST.append(acc)
    ACC_LIST = np.array(ACC_LIST)
    LANE_CHANGE_INDEX_LIST = [0, 1, 2]
    NDD_ACC_MIN = global_val.acc_low
    NDD_ACC_MAX = global_val.acc_high
    # Default longitudinal policy parameters (use IDM when naturalistic data is not available)
    COMFORT_ACC_MAX = 2.0  # [m/s2]  2
    COMFORT_ACC_MIN = -4.0  # [m/s2]  -4
    DISTANCE_WANTED = 5.0  # [m]  5
    TIME_WANTED = 1.5  # [s]  1.5
    DESIRED_VELOCITY = 35  # [m/s]
    DELTA = 4.0  # []

    # Default lateral policy parameters (use MOBIL when naturalistic data is not available)
    POLITENESS = 0.5  # in [0, 1]  0.5
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]  0.2
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 3.0  # [m/s2]  3
    LANE_CHANGE_DELAY = 1.0  # [s]

    def __init__(self, road, position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=None,
                 route=None,
                 enable_lane_change=True,
                 timer=None):
        super(IDMVehicle, self).__init__(road, position, heading, velocity, target_lane_index, target_velocity, route)
        self.enable_lane_change = enable_lane_change
        self.IDM_flag = False
        self.timer = timer or (np.sum(self.position) * np.pi) % self.LANE_CHANGE_DELAY
        self.action_mode = "DriverModel"  # NDD or DriverModel (IDM+MOBIL)
        self.longi_acc = 0

    def randomize_behavior(self):
        pass

    @classmethod
    def create_from(cls, vehicle):
        """
            Create a new vehicle from an existing one.
            The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, velocity=vehicle.velocity,
                target_lane_index=vehicle.target_lane_index, target_velocity=vehicle.target_velocity,
                route=vehicle.route, timer=getattr(vehicle, 'timer', None))
        return v

    def act(self, bv_action=None):
        """
            Execute an action.

            For now, no action is supported because the vehicle takes all decisions
            of acceleration and lane changes on its own, based on the IDM and MOBIL models.

        :param action: the action
        """
        if self.crashed:
            return
        if bv_action or self.controlled:
            self.IDM_flag = False
            self.follow_road()
            _from, _to, _id = self.lane_index
            if bv_action:
                self.actual_action = bv_action
                if bv_action == "LANE_RIGHT" and self.lane_index == len(self.road.network.graph[_from][_to]) - 1:
                    self.actual_action = "IDLE"
                elif bv_action == "LANE_LEFT" and self.lane_index == 0:
                    self.actual_action = "IDLE"

            if bv_action == "LANE_RIGHT":
                self.longi_acc = 0
                _from, _to, _id = self.target_lane_index
                target_lane_index = _from, _to, np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1)
                if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                    self.target_lane_index = target_lane_index
            elif bv_action == "LANE_LEFT":
                self.longi_acc = 0
                _from, _to, _id = self.target_lane_index
                target_lane_index = _from, _to, np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1)
                if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                    self.target_lane_index = target_lane_index
            elif bv_action == "IDLE":
                self.longi_acc = 0
            elif bv_action:
                self.longi_acc = int(bv_action)
            action = {'steering': self.steering_control(self.target_lane_index),
                      'acceleration': self.longi_acc}
            super(ControlledVehicle, self).act(action)
            return

        action = {}
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)

        if self.action_mode == "DriverModel":
            self.IDM_flag = True
            # Lateral: MOBIL
            self.follow_road()
            if self.enable_lane_change:
                lane_change_flag, _ = self.change_lane_policy()
            action['steering'] = self.steering_control(self.target_lane_index)

            # Longitudinal: IDM
            # action['acceleration'] = self.acceleration(ego_vehicle=self,front_vehicle=front_vehicle,rear_vehicle=rear_vehicle)
            tmp_acc = self.acceleration(ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle)
            tmp_acc = np.clip(tmp_acc, global_val.acc_low, global_val.acc_high)
            acc_possi_list = scipy.stats.norm.pdf(self.ACC_LIST, tmp_acc, 0.3)
            acc_possi_list = acc_possi_list / (sum(acc_possi_list))
            # self.longi_acc = np.random.normal(, 0.2, None)
            action['acceleration'] = np.random.choice(self.ACC_LIST, None, False, acc_possi_list)
            super(ControlledVehicle, self).act(action)
            self.longi_acc = action['acceleration']
            return action['acceleration'], acc_possi_list

    def step(self, dt):
        """
            Step the simulation.

            Increases a timer used for decision policies, and step the vehicle dynamics.

        :param dt: timestep
        """
        self.timer += dt
        super(IDMVehicle, self).step(dt)

    def acceleration(self, ego_vehicle, front_vehicle=None, rear_vehicle=None):
        """
            Compute an acceleration command with the Intelligent Driver Model.

            The acceleration is chosen so as to:
            - reach a target velocity;
            - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        if not ego_vehicle:
            return 0
        # acceleration = self.COMFORT_ACC_MAX * (
        #         1 - np.power(ego_vehicle.velocity / utils.not_zero(ego_vehicle.target_velocity), self.DELTA))
        acceleration = self.COMFORT_ACC_MAX * (
                1 - np.power(ego_vehicle.velocity / self.DESIRED_VELOCITY, self.DELTA))
        if front_vehicle:
            d = max(1e-5, ego_vehicle.lane_distance_to(front_vehicle) - self.LENGTH)
            acceleration -= self.COMFORT_ACC_MAX * \
                            np.power(self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2)
        return acceleration

    def desired_gap(self, ego_vehicle, front_vehicle=None):
        """
            Compute the desired distance between a vehicle and its leading vehicle.

        :param ego_vehicle: the vehicle being controlled
        :param front_vehicle: its leading vehicle
        :return: the desired distance between the two [m]
        """
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = ego_vehicle.velocity - front_vehicle.velocity
        d_star = d0 + max(0, ego_vehicle.velocity * tau + ego_vehicle.velocity * dv / (2 * np.sqrt(ab)))
        return d_star

    def maximum_velocity(self, front_vehicle=None):
        """
            Compute the maximum allowed velocity to avoid Inevitable Collision States.

            Assume the front vehicle is going to brake at full deceleration and that
            it will be noticed after a given delay, and compute the maximum velocity
            which allows the ego-vehicle to brake enough to avoid the collision.

        :param front_vehicle: the preceding vehicle
        :return: the maximum allowed velocity, and suggested acceleration
        """
        if not front_vehicle:
            return self.target_velocity
        d0 = self.DISTANCE_WANTED
        a0 = self.COMFORT_ACC_MIN
        a1 = self.COMFORT_ACC_MIN
        tau = self.TIME_WANTED
        d = max(self.lane_distance_to(front_vehicle) - self.LENGTH / 2 - front_vehicle.LENGTH / 2 - d0, 0)
        v1_0 = front_vehicle.velocity
        delta = 4 * (a0 * a1 * tau) ** 2 + 8 * a0 * (a1 ** 2) * d + 4 * a0 * a1 * v1_0 ** 2
        v_max = -a0 * tau + np.sqrt(delta) / (2 * a1)

        # Velocity control
        self.target_velocity = min(self.maximum_velocity(front_vehicle), self.target_velocity)
        acceleration = self.velocity_control(self.target_velocity)

        return v_max, acceleration

    def change_lane_policy(self, modify_flag=True):
        """
        Decide when to change lane.

        Based on:
        - frequency;
        - closeness of the target lane;
        - MOBIL model.

        When modify_flag is False, it just predict the lane change decision of the CAV and not do the real control of the CAV
        """
        to_lane_id = None

        # If a lane change already ongoing
        if self.lane_index != self.target_lane_index:
            # If we are on correct route but bad lane: abort it if someone else is already changing into the same lane
            if self.lane_index[:2] == self.target_lane_index[:2]:
                for v in self.road.vehicles:
                    if v is not self \
                            and v.lane_index != self.target_lane_index \
                            and isinstance(v, ControlledVehicle) \
                            and v.target_lane_index == self.target_lane_index:
                        d = self.lane_distance_to(v)
                        d_star = self.desired_gap(self, v)
                        if 0 < d < d_star:
                            # print("self_index:",self.lane_index)
                            # print("change_index:",lane_index)
                            self.target_lane_index = self.lane_index
                            break
            return False, to_lane_id

        # else, at a given frequency,
        if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer):
            return False, to_lane_id
        if modify_flag:
            self.timer = 0

        # decide to make a lane change
        for lane_index in self.road.network.side_lanes(self.lane_index):
            # Is the candidate lane close enough?

            if not self.road.network.get_lane(lane_index).is_reachable_from(self.position):
                continue
            # Does the MOBIL model recommend a lane change?
            if self.mobil(lane_index):
                # print("self_index:",self.lane_index)
                # print("change_index:",lane_index)
                if modify_flag:
                    self.target_lane_index = lane_index
                to_lane_id = lane_index[2]
                return True, to_lane_id
        return False, to_lane_id

    def mobil(self, lane_index):
        """
            MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change
            The vehicle should change lane only if:
            - after changing it (and/or following vehicles) can accelerate more;
            - it doesn't impose an unsafe braking on its new following vehicle.

        :param lane_index: the candidate lane for the change
        :return: whether the lane change should be performed
        """
        # Is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = self.road.neighbour_vehicles(self, lane_index)
        new_following_a = self.acceleration(ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = self.acceleration(ego_vehicle=new_following, front_vehicle=self)
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Do I have a planned route for a specific lane which is safe for me to access?
        old_preceding, old_following = self.road.neighbour_vehicles(self)
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        if self.route and self.route[0][2]:
            # Wrong direction
            if np.sign(lane_index[2] - self.target_lane_index[2]) != np.sign(self.route[0][2] - self.target_lane_index[2]):
                return False
            # Unsafe braking required
            elif self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False

        # Is there an acceleration advantage for me and/or my followers to change lane?
        else:
            self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
            old_following_a = self.acceleration(ego_vehicle=old_following, front_vehicle=self)
            old_following_pred_a = self.acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
            jerk = self_pred_a - self_a + self.POLITENESS * (new_following_pred_a - new_following_a + old_following_pred_a - old_following_a)
            if jerk <= self.LANE_CHANGE_MIN_ACC_GAIN:
                return False

        return True

    def recover_from_stop(self, acceleration):
        """
            If stopped on the wrong lane, try a reversing maneuver.

        :param acceleration: desired acceleration from IDM
        :return: suggested acceleration to recover from being stuck
        """
        stopped_velocity = 5
        safe_distance = 200
        # Is the vehicle stopped on the wrong lane?
        if self.target_lane_index != self.lane_index and self.velocity < stopped_velocity:
            _, rear = self.road.neighbour_vehicles(self)
            _, new_rear = self.road.neighbour_vehicles(self, self.road.network.get_lane(self.target_lane_index))
            # Check for free room behind on both lanes
            if (not rear or rear.lane_distance_to(self) > safe_distance) and \
                    (not new_rear or new_rear.lane_distance_to(self) > safe_distance):
                # Reverse
                return -self.COMFORT_ACC_MAX / 2
        return acceleration


class NDDVehicle(IDMVehicle):
    """
    A vehicle using both a longitudinal and a lateral decision policies. The stochastic behavior models (both longitudinal and lateral)
    are developed based on naturalistic driving data. For states that not covered by naturalistic data, a IDM and MOBIL model will be used.

    """
    mode = np.nan  # In the current time step, this veh is doing CF (car-following) or FF (free-flow) or LC (lane-change) or IDM (using IDM and MOBIL).
    v, x, lane_idx, r, rr = np.nan, np.nan, np.nan, np.nan, np.nan  # The velocity, range, range rate of the vehicle at the specific timestamp when doing the decision.
    round_r, round_rr, round_v = np.nan, np.nan, np.nan
    pdf_distribution, ndd_possi = np.nan, np.nan  # In this timestamp, the action pdf distribution and the probability of choosing the current action
    LC_related = np.nan

    mode_prev = np.nan  # The state of the previous time step.
    v_prev, x_prev, lane_idx_prev, r_prev, rr_prev = np.nan, np.nan, np.nan, np.nan, np.nan  # The velocity, range, range rate of the vehicle at the specific timestamp when doing the decision.
    round_r_prev, round_rr_prev, round_v_prev = np.nan, np.nan, np.nan
    pdf_distribution_prev, ndd_possi_prev = np.nan, np.nan
    longi_acc_prev = np.nan
    LC_related_prev = np.nan

    def get_Lateral_NDD_possi_list(self):
        """
        Get current BV lateral behavior distribution.

        Returns:
            np.array of action probability mass function - [Left turn probability, Go straight probability, Right turn probability].
        """
        lane_id = self.lane_index[2]
        observation = []  # obervation for this BV
        if lane_id == 0:
            f0, r0 = None, None
            f1, r1 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 0), obs_range=global_val.bv_obs_range)
            f2, r2 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 1), obs_range=global_val.bv_obs_range)
            observation = [f1, r1, f0, r0, f2, r2]
        if lane_id == 1:
            f0, r0 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 0), obs_range=global_val.bv_obs_range)
            f1, r1 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 1), obs_range=global_val.bv_obs_range)
            f2, r2 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 2), obs_range=global_val.bv_obs_range)
            observation = [f1, r1, f0, r0, f2, r2]
        if lane_id == 2:
            f0, r0 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 1), obs_range=global_val.bv_obs_range)
            f1, r1 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 2), obs_range=global_val.bv_obs_range)
            f2, r2 = None, None
            observation = [f1, r1, f0, r0, f2, r2]
        _, _, lane_change_pdf_array = self.Lateral_NDD(observation, modify_flag=False)
        return lane_change_pdf_array

    def get_Longi_NDD_possi_list(self):
        """
        Get current BV longitudinal behavior distribution.

        Returns:
            np.array of action probability mass function.
        """
        lane_id = self.lane_index[2]
        observation = []  # obervation for this BV
        if lane_id == 0:
            f0, r0 = None, None
            f1, r1 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 0), obs_range=global_val.bv_obs_range)
            f2, r2 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 1), obs_range=global_val.bv_obs_range)
            observation = [f1, r1, f0, r0, f2, r2]
        if lane_id == 1:
            f0, r0 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 0), obs_range=global_val.bv_obs_range)
            f1, r1 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 1), obs_range=global_val.bv_obs_range)
            f2, r2 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 2), obs_range=global_val.bv_obs_range)
            observation = [f1, r1, f0, r0, f2, r2]
        if lane_id == 2:
            f0, r0 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 1), obs_range=global_val.bv_obs_range)
            f1, r1 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 2), obs_range=global_val.bv_obs_range)
            f2, r2 = None, None
            observation = [f1, r1, f0, r0, f2, r2]
        _, possi_list = self.Longitudinal_NDD(observation, modify_flag=False)
        return possi_list

    def act(self, bv_action=None, essential_flag=False, enable_NDD_lane_change_flag=True):
        """
        Execute an action. If given action, then execute it. Otherwise, use naturalistic driving model or IDM+MOBIL (when
        naturalistic driving models are not available).

        Args:
            bv_action: the given action.
            essential_flag: 0 means current moment is doing decision making to choose new action. Otherwise, continue previous action.
            enable_NDD_lane_change_flag: whether allow lane change.

        """
        action = {}
        if self.crashed:
            return

        if essential_flag == 0:  # Current moment is doing decision making.
            # if self.mode != "IDM":
            self.mode_prev, self.pdf_distribution_prev = copy.deepcopy(self.mode), copy.deepcopy(self.pdf_distribution)
            self.v_prev, self.r_prev, self.rr_prev, self.round_v_prev, self.round_r_prev, self.round_rr_prev, self.ndd_possi_prev, self.longi_acc_prev = copy.deepcopy(
                self.v), copy.deepcopy(self.r), copy.deepcopy(self.rr), copy.deepcopy(self.round_v), copy.deepcopy(self.round_r), copy.deepcopy(self.round_rr), copy.deepcopy(
                self.ndd_possi), copy.deepcopy(self.longi_acc)
            self.LC_related_prev = copy.deepcopy(self.LC_related)
            self.x_prev, self.lane_idx_prev = copy.deepcopy(self.x), copy.deepcopy(self.lane_idx)

            self.mode, self.pdf_distribution = np.nan, np.nan
            self.v, self.x, self.lane_idx = copy.deepcopy(self.velocity), copy.deepcopy(self.position[0]), copy.deepcopy(self.lane_index[2])
            self.r, self.rr, self.round_v, self.round_r, self.round_rr, self.ndd_possi, self.LC_related = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

            self.IDM_flag = False  # Reset the veh flag to the default NDD model flag
            lane_id = self.lane_index[2]
            observation = self._get_veh_obs()
            if bv_action:
                self.follow_road()
                _from, _to, _id = self.lane_index
                if bv_action:
                    self.actual_action = bv_action
                    if bv_action == "LANE_RIGHT" and self.lane_index == len(self.road.network.graph[_from][_to]) - 1:
                        self.actual_action = "IDLE"
                        self.mode = "Controlled-Should not here!"
                    elif bv_action == "LANE_LEFT" and self.lane_index == 0:
                        self.actual_action = "IDLE"
                        self.mode = "Controlled-Should not here!"

                if bv_action == "LANE_RIGHT":
                    self.longi_acc = 0
                    _from, _to, _id = self.target_lane_index
                    target_lane_index = _from, _to, np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1)
                    if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                        self.target_lane_index = target_lane_index
                    self.mode = "Controlled-Right"
                elif bv_action == "LANE_LEFT":
                    self.longi_acc = 0
                    _from, _to, _id = self.target_lane_index
                    target_lane_index = _from, _to, np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1)
                    if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                        self.target_lane_index = target_lane_index
                    self.mode = "Controlled-Left"
                elif bv_action:
                    self.mode = "Controlled-Long"
                    self.longi_acc = float(bv_action)
                _, _, lane_change_pdf_array = self.Lateral_NDD(observation, modify_flag=False)
                _, longi_pdf_array = self.Longitudinal_NDD(observation)
                action = {'steering': self.steering_control(self.target_lane_index), 'acceleration': self.longi_acc}
                super(ControlledVehicle, self).act(action)
                ndd_possi = 0
                if self.actual_action == "LANE_LEFT":
                    ndd_possi = lane_change_pdf_array[0]
                elif self.actual_action == "LANE_RIGHT":
                    ndd_possi = lane_change_pdf_array[2]
                else:
                    acc_idx = list(self.ACC_LIST).index(self.longi_acc)
                    ndd_possi = longi_pdf_array[acc_idx] * lane_change_pdf_array[1]
                return action, ndd_possi, None

            else:
                # Lateral: NDD
                self.follow_road()
                lane_change_flag, lane_change_idx, lane_change_pdf_array = False, 1, np.array([0, 1, 0])
                if self.enable_lane_change and enable_NDD_lane_change_flag:
                    lane_change_flag, lane_change_idx, lane_change_pdf_array = self.Lateral_NDD(observation)
                    # print(lane_change_pdf_array)
                if not global_val.enable_One_lead_LC and not global_val.enable_Single_LC and not global_val.enable_Double_LC and not enable_NDD_lane_change_flag:
                    assert (not lane_change_flag)
                # Longitudinal: NDD
                if not lane_change_flag:
                    self.longi_acc, longi_pdf_array = self.Longitudinal_NDD(observation, modify_flag=True)
                else:
                    # _, longi_pdf_array = self.Longitudinal_NDD(observation)
                    self.longi_acc = 0

        action['acceleration'] = self.longi_acc
        action['steering'] = self.steering_control(self.target_lane_index)
        super(ControlledVehicle, self).act(action)
        if essential_flag == 0:
            ndd_possi = 0
            if lane_change_idx == 0:
                ndd_possi = lane_change_pdf_array[0]
            elif lane_change_idx == 2:
                ndd_possi = lane_change_pdf_array[2]
            else:
                acc_idx = list(self.ACC_LIST).index(self.longi_acc)
                ndd_possi = longi_pdf_array[acc_idx] * lane_change_pdf_array[1]
            self.ndd_possi = ndd_possi
        else:
            ndd_possi = None
        return action, ndd_possi, None

    def round_value_lane_change(self, real_value, value_list, round_item=None):
        if real_value < value_list[0]:
            real_value = value_list[0]
        elif real_value > value_list[-1]:
            real_value = value_list[-1]

        if global_val.round_rule == "Round_to_closest":
            min_val, max_val, resolution = value_list[0], value_list[-1], value_list[1] - value_list[0]
            real_value = np.clip(round((real_value - (min_val)) / resolution) * resolution + (min_val), min_val, max_val)

        if round_item == "speed":
            value_idx = bisect.bisect_left(value_list, real_value)
            value_idx = value_idx if real_value <= value_list[-1] else value_idx - 1
            assert value_idx <= (len(value_list) - 1)
            assert value_idx >= 0
            round_value = value_list[value_idx]
            return round_value, value_idx
        else:
            value_idx = bisect.bisect_left(value_list, real_value)
            value_idx = value_idx - 1 if real_value != value_list[value_idx] else value_idx
            assert value_idx <= (len(value_list) - 1)
            assert value_idx >= 0
            round_value = value_list[value_idx]
            return round_value, value_idx

    def _check_bound_constraints(self, value, bound_low, bound_high):
        if value < bound_low or value > bound_high:
            return False
        else:
            return True

    def _get_One_lead_LC_prob(self, veh_front):
        """
        Lane change situation that no vehicles on the target lane.

        Args:
            veh_front: vehicle in front of the subject vehicle on the same lane.

        Returns:
            (float, tuple): lane change probability to the target lane, lane change related information.
        """

        lane_id, x, v = self.lane_index[2], self.position[0], self.velocity
        LC_related = None
        if not global_val.enable_One_lead_LC:
            return 0, LC_related
        r, rr = veh_front.position[0] - x - self.LENGTH, veh_front.velocity - v
        # Check bound
        if not self._check_bound_constraints(v, global_val.lc_v_low, global_val.lc_v_high):
            return 0, LC_related
        if not self._check_bound_constraints(r, global_val.lc_rf_low, global_val.lc_rf_high):
            return 0, LC_related
        if not self._check_bound_constraints(rr, global_val.lc_rrf_low, global_val.lc_rrf_high):
            return 0, LC_related

        round_r, round_r_idx = self.round_value_lane_change(real_value=r, value_list=global_val.lc_r1_list)
        round_rr, round_rr_idx = self.round_value_lane_change(real_value=rr, value_list=global_val.lc_rr1_list)
        round_speed, round_speed_idx = self.round_value_lane_change(real_value=v, value_list=global_val.lc_v_list, round_item="speed")
        assert ((round_r - 1) <= r <= (round_r + 1) and (round_rr - 1) <= rr <= (round_rr + 1))

        lane_change_prob = global_val.OL_pdf[round_speed_idx, round_r_idx, round_rr_idx, :][0]
        LC_related = (v, r, rr, round_speed, round_r, round_rr)

        return lane_change_prob, LC_related

    def _get_Double_LC_prob(self, veh_adj_front, veh_adj_rear):
        """
        Lane change situation that are vehicles both in front and behind the subject vehicle on the target lane.

        Args:
            veh_adj_front: vehicle in front of the subject vehicle on the target lane.
            veh_adj_rear: vehicle behind the subject vehicle on the target lane.

        Returns:
            (float, tuple): lane change probability to the target lane, lane change related information.

        """
        lane_id, x, v = self.lane_index[2], self.position[0], self.velocity

        LC_related = None
        # Double lane change
        if not global_val.enable_Double_LC:
            return 0, LC_related
        r1, rr1 = veh_adj_front.position[0] - x - self.LENGTH, veh_adj_front.velocity - v
        r2, rr2 = x - veh_adj_rear.position[0] - self.LENGTH, v - veh_adj_rear.velocity
        if not self._check_bound_constraints(v, global_val.lc_v_low, global_val.lc_v_high):
            return 0, LC_related
        elif not self._check_bound_constraints(r1, global_val.lc_rf_low, global_val.lc_rf_high):
            return 0, LC_related
        elif not self._check_bound_constraints(rr1, global_val.lc_rrf_low, global_val.lc_rrf_high):
            return 0, LC_related
        elif not self._check_bound_constraints(r2, global_val.lc_re_low, global_val.lc_re_high):
            return 0, LC_related
        elif not self._check_bound_constraints(rr2, global_val.lc_rre_low, global_val.lc_rre_high):
            return 0, LC_related
        round_v, v_idx = self.round_value_lane_change(real_value=v, value_list=global_val.lc_v_list, round_item="speed")
        round_r1, r1_idx = self.round_value_lane_change(real_value=r1, value_list=global_val.lc_r1_list)
        round_rr1, rr1_idx = self.round_value_lane_change(real_value=rr1, value_list=global_val.lc_rr1_list)
        round_r2, r2_idx = self.round_value_lane_change(real_value=r2, value_list=global_val.lc_r2_list)
        round_rr2, rr2_idx = self.round_value_lane_change(real_value=rr2, value_list=global_val.lc_rr2_list)

        lane_change_prob = global_val.DLC_pdf[v_idx, r1_idx, rr1_idx, r2_idx, rr2_idx, :][0]

        LC_related = (v, r1, rr1, r2, rr2, round_v, round_r1, round_rr1, round_r2, round_rr2, lane_change_prob)
        return lane_change_prob, LC_related

    def _get_Single_LC_prob(self, veh_front, veh_adj_front):
        """
        Lane change situation that there is a vehicle in front of the subject vehicle on the target lane.

        Args:
            veh_front: vehicle in front of the subject vehicle on the same lane.
            veh_adj_front: vehicle in front of the subject vehicle on the target lane.

        Returns:
            (float, tuple): lane change probability to the target lane, lane change related information.

        """
        lane_id, x, v = self.lane_index[2], self.position[0], self.velocity

        LC_related = None
        # Single lane change
        if not global_val.enable_Single_LC:
            return 0, LC_related

        r1, rr1 = veh_front.position[0] - x - self.LENGTH, veh_front.velocity - v
        r2, rr2 = veh_adj_front.position[0] - x - self.LENGTH, veh_adj_front.velocity - v

        if not self._check_bound_constraints(v, global_val.lc_v_low, global_val.lc_v_high):
            return 0, LC_related
        elif not self._check_bound_constraints(r1, global_val.lc_rf_low, global_val.lc_rf_high):
            return 0, LC_related
        elif not self._check_bound_constraints(rr1, global_val.lc_rrf_low, global_val.lc_rrf_high):
            return 0, LC_related
        elif not self._check_bound_constraints(r2, global_val.lc_rf_low, global_val.lc_rf_high):
            return 0, LC_related
        elif not self._check_bound_constraints(rr2, global_val.lc_rrf_low, global_val.lc_rrf_high):
            return 0, LC_related

        round_v, v_idx = self.round_value_lane_change(real_value=v, value_list=global_val.lc_v_list, round_item="speed")
        round_r1, r1_idx = self.round_value_lane_change(real_value=r1, value_list=global_val.lc_r1_list)
        round_rr1, rr1_idx = self.round_value_lane_change(real_value=rr1, value_list=global_val.lc_rr1_list)
        round_r2, r2_idx = self.round_value_lane_change(real_value=r2, value_list=global_val.lc_r2_list)
        round_rr2, rr2_idx = self.round_value_lane_change(real_value=rr2, value_list=global_val.lc_rr2_list)

        lane_change_prob = global_val.SLC_pdf[v_idx, r1_idx, rr1_idx, r2_idx, rr2_idx, :][0]

        LC_related = (v, r1, rr1, r2, rr2, round_v, round_r1, round_rr1, round_r2, round_rr2, lane_change_prob)
        return lane_change_prob, LC_related

    def _get_Cut_in_LC_prob(self, veh_front, veh_adj_rear):
        """
        Lane change situation that there is a vehicle behind the subject vehicle on the target lane.

        Args:
            veh_front: vehicle in front of the subject vehicle on the same lane.
            veh_adj_rear: vehicle behind the subject vehicle on the same lane.

        Returns:
            (float, tuple): lane change probability to the target lane, lane change related information.

        """
        lane_id, x, v = self.lane_index[2], self.position[0], self.velocity

        LC_related = None

        if not global_val.enable_Cut_in_LC:
            return 0, None

        r1, rr1 = veh_front.position[0] - x - self.LENGTH, veh_front.velocity - v
        r2, rr2 = x - veh_adj_rear.position[0] - self.LENGTH, v - veh_adj_rear.velocity

        if not self._check_bound_constraints(v, global_val.lc_v_low, global_val.lc_v_high):
            return 0, LC_related
        elif not self._check_bound_constraints(r1, global_val.lc_rf_low, global_val.lc_rf_high):
            return 0, LC_related
        elif not self._check_bound_constraints(rr1, global_val.lc_rrf_low, global_val.lc_rrf_high):
            return 0, LC_related
        elif not self._check_bound_constraints(r2, global_val.lc_rf_low, global_val.lc_rf_high):
            return 0, LC_related
        elif not self._check_bound_constraints(rr2, global_val.lc_rrf_low, global_val.lc_rrf_high):
            return 0, LC_related

        round_v, v_idx = self.round_value_lane_change(real_value=v, value_list=global_val.lc_v_list, round_item="speed")
        round_r1, r1_idx = self.round_value_lane_change(real_value=r1, value_list=global_val.lc_r1_list)
        round_rr1, rr1_idx = self.round_value_lane_change(real_value=rr1, value_list=global_val.lc_rr1_list)
        round_r2, r2_idx = self.round_value_lane_change(real_value=r2, value_list=global_val.lc_r2_list)
        round_rr2, rr2_idx = self.round_value_lane_change(real_value=rr2, value_list=global_val.lc_rr2_list)

        lane_change_prob = global_val.CI_pdf[v_idx, r1_idx, rr1_idx, r2_idx, rr2_idx, :][0]

        LC_related = (v, r1, rr1, r2, rr2, round_v, round_r1, round_rr1, round_r2, round_rr2, lane_change_prob)
        return lane_change_prob, LC_related

    def _LC_prob(self, surrounding_vehicles, full_obs):
        """
        Given surrounding vehicles (current lane leading vehicle, target lane leading and following vehicles if applicable) information,
        determine the lane-change probability according to the naturalistic behavior models.

        Args:
            surrounding_vehicles: surrounding vehicles information (current lane leading vehicle, target lane leading and following vehicles if applicable).
            full_obs: observations include other surrounding vehicles.

        Returns:
            (float, string, tuple): Lane change probability, lane change category, lane change related information.
        """
        LC_prob, E_LC_prob = None, None
        veh_front, veh_adj_front, veh_adj_rear = surrounding_vehicles

        if not veh_adj_front and not veh_adj_rear:
            # One lead LC
            LC_prob, LC_related = self._get_One_lead_LC_prob(veh_front)
            E_LC_prob = LC_prob
            return E_LC_prob, "One_lead", LC_related

        elif veh_adj_front and not veh_adj_rear:
            # Single lane change
            LC_prob, LC_related = self._get_Single_LC_prob(veh_front, veh_adj_front)
            E_LC_prob = LC_prob
            return E_LC_prob, "SLC", LC_related

        elif not veh_adj_front and veh_adj_rear:
            # OL prob
            OL_LC_prob, OL_LC_related = self._get_One_lead_LC_prob(veh_front)

            # CI prob
            CI_LC_prob, CI_LC_related = self._get_Cut_in_LC_prob(veh_front, veh_adj_rear)
            LC_related = CI_LC_related

            r_adj = self.position[0] - veh_adj_rear.position[0] - self.LENGTH

            if r_adj >= global_val.min_r_ignore:
                E_LC_prob = global_val.ignore_adj_veh_prob * OL_LC_prob + (1 - global_val.ignore_adj_veh_prob) * CI_LC_prob
            else:
                E_LC_prob = CI_LC_prob
            return E_LC_prob, "Cut_in", LC_related

        elif veh_adj_front and veh_adj_rear:
            # SLC prob
            SLC_LC_prob, SLC_LC_related = self._get_Single_LC_prob(veh_front, veh_adj_front)

            # DLC prob
            DLC_LC_prob, DLC_LC_related = self._get_Double_LC_prob(veh_adj_front, veh_adj_rear)
            LC_related = DLC_LC_related

            r_adj = self.position[0] - veh_adj_rear.position[0] - self.LENGTH

            if r_adj >= global_val.min_r_ignore:
                E_LC_prob = global_val.ignore_adj_veh_prob * SLC_LC_prob + (1 - global_val.ignore_adj_veh_prob) * DLC_LC_prob
            else:
                E_LC_prob = DLC_LC_prob
            return E_LC_prob, "DLC", LC_related

    def Lateral_NDD(self, obs, modify_flag=True):
        """
        Determine the lateral behavior of the vehicle.

        Args:
            obs: vehicle observation.
            modify_flag: whether do real control. Since this function might be used to get vehicle naturalistic action probability.

        Returns:
            - lane change flag - If True, then do lane change.
            - lane_change_idx - 0:Turn Left, 1:Go Straight, 2: Turn Right.
            - action probability mass function - [Left turn probability, Go straight probability, Right turn probability].
        """
        initial_pdf = np.array([0, 1, 0])  # Left, Still, Right
        if not isinstance(global_val.SLC_pdf, np.ndarray) or not isinstance(global_val.DLC_pdf, np.ndarray) or not isinstance(global_val.OL_pdf, np.ndarray) or not isinstance(global_val.CI_pdf, np.ndarray):
            raise ValueError("No naturalistic driving models input!")

        lane_id, x, v = self.lane_index[2], self.position[0], self.velocity
        f1, r1, f0, r0, f2, r2 = obs

        if not f1:  # No vehicle ahead
            return False, 1, initial_pdf
        else:  # Has vehicle ahead
            left_prob, still_prob, right_prob = 0, 0, 0
            LC_related_list = []
            LC_type_list = []

            # Check whether there is CF data in this situation, if not then use IDM and MOBIL.
            has_CF_data_flag = self.check_whether_has_CF_data(f1)
            MOBIL_flag = (not has_CF_data_flag) and (np.floor(v + 0.5) <= 21)

            # MOBIL
            if MOBIL_flag:
                left_prob, right_prob = self.MOBIL_result()

                LC_related_list = [(v), (v)]
                LC_type_list = ["MOBIL", "MOBIL"]

            # NDD
            else:
                for item in ["Left", "Right"]:
                    if item == "Left":
                        surrounding = (f1, f0, r0)
                        left_prob, LC_type, LC_related = self._LC_prob(surrounding, obs)
                        LC_related_list.append(LC_related)
                        LC_type_list.append(LC_type)
                    else:
                        surrounding = (f1, f2, r2)
                        right_prob, LC_type, LC_related = self._LC_prob(surrounding, obs)
                        LC_related_list.append(LC_related)
                        LC_type_list.append(LC_type)

            if lane_id == 0:
                left_prob = 0
            if lane_id == 2:
                right_prob = 0
            if left_prob + right_prob > 1:
                tmp = left_prob + right_prob
                left_prob *= 0.9 / (tmp)
                right_prob *= 0.9 / (tmp)
            still_prob = 1 - left_prob - right_prob
            pdf_array = np.array([left_prob, still_prob, right_prob])

            lane_change_idx = np.random.choice(self.LANE_CHANGE_INDEX_LIST, None, False, pdf_array)
            to_lane_id = lane_id + lane_change_idx - 1
            if lane_change_idx != 1:
                if modify_flag:
                    self.target_lane_index = ("a", "b", to_lane_id)
                    if lane_change_idx == 0:
                        self.mode = LC_type_list[0]
                        self.LC_related = LC_related_list[0]
                        # print("LLC", self.mode, self.LC_related, pdf_array)
                    elif lane_change_idx == 2:
                        self.mode = LC_type_list[1]
                        self.LC_related = LC_related_list[1]
                        # print("RLC", self.mode, self.LC_related, pdf_array)
                return True, lane_change_idx, pdf_array
            else:
                return False, lane_change_idx, pdf_array

    def round_value_function(self, real_value, round_item):
        """
        Round the data.

        Args:
            real_value: current continuous data value.
            round_item: the item is rounding, e.g., velocity, range, range rate.

        Returns:
            (float, int): round value and the index of the round value among the state space.
        """
        if round_item == "speed":
            value_list = global_val.speed_list
            value_dic = global_val.v_to_idx_dic
            min_val, max_val, resolution = global_val.v_low, global_val.v_high, global_val.v_step
        elif round_item == "range":
            value_list = global_val.r_list
            value_dic = global_val.r_to_idx_dic
            min_val, max_val, resolution = global_val.r_low, global_val.r_high, global_val.r_step
        elif round_item == "range_rate":
            value_list = global_val.rr_list
            value_dic = global_val.rr_to_idx_dic
            min_val, max_val, resolution = global_val.rr_low, global_val.rr_high, global_val.rr_step

        if real_value < value_list[0]:
            real_value = value_list[0]
        elif real_value > value_list[-1]:
            real_value = value_list[-1]

        if global_val.round_rule == "Round_to_closest":
            round_tmp = np.clip(round((real_value - (min_val)) / resolution) * resolution + (min_val), min_val, max_val)
            try:
                assert (-0.5 < round_tmp - real_value <= 0.5)
            except:
                round_tmp += 1
                try:
                    assert (-0.5 < round_tmp - real_value <= 0.5)
                except:
                    print(real_value, round_tmp, round_item)
                    raise ValueError("ROUND!")

        value_idx = bisect.bisect_left(value_list, round_tmp)
        value_idx = value_idx - 1 if abs(round_tmp - value_list[value_idx]) > 1e-5 else value_idx
        assert (abs(round_tmp - value_list[value_idx]) < 1e-5)
        round_value = value_list[value_idx]
        assert (value_dic[round_value] == value_idx)
        assert (-0.5 < round_value - real_value <= 0.5)
        return round_value, value_idx

    def _MOBIL_model(self, lane_index):
        """
        Use a MOBIL model when there is no naturalistic driving data.

        Args:
            lane_index: the candidate lane for the change

        Returns:
            (bool, float):
            - lane change flag: whether to perform the LC.
            - gain: the gain of performing LC.

        """
        gain = None

        # Is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = self.road.neighbour_vehicles(self, lane_index)

        # Check whether will crash immediately
        r_new_preceding, r_new_following = 99999, 99999
        if new_preceding:
            r_new_preceding = new_preceding.position[0] - self.position[0] - self.LENGTH
        if new_following:
            r_new_following = self.position[0] - new_following.position[0] - self.LENGTH
        if r_new_preceding <= 0 or r_new_following <= 0:
            return False, gain

        new_following_a = self.acceleration(ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = self.acceleration(ego_vehicle=new_following, front_vehicle=self)

        # The deceleration of the new following vehicle after the the LC should not be too big (negative)
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False, gain

        old_preceding, old_following = self.road.neighbour_vehicles(self)
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)

        # Is there an acceleration advantage for me and/or my followers to change lane?
        self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
        old_following_a = self.acceleration(ego_vehicle=old_following, front_vehicle=self)
        old_following_pred_a = self.acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
        gain = self_pred_a - self_a + self.POLITENESS * (new_following_pred_a - new_following_a + old_following_pred_a - old_following_a)
        if gain <= self.LANE_CHANGE_MIN_ACC_GAIN:
            return False, gain
        return True, gain

    def MOBIL_result(self):
        """
        Use a MOBIL model when there is no naturalistic driving data.

        Returns:
            (float, float): left and right lane change probability.
        """
        left_prob, right_prob = 0, 0
        lane_id = self.lane_index[2]

        MOBIL_LC_prob = 1
        left_gain, right_gain = -np.inf, -np.inf
        left_LC_flag, right_LC_flag = False, False
        for lane_index in self.road.network.side_lanes(self.lane_index):
            LC_flag, gain = self._MOBIL_model(lane_index)
            if LC_flag:
                if lane_index[2] > lane_id:
                    right_LC_flag, right_gain = LC_flag, gain
                elif lane_index[2] < lane_id:
                    left_LC_flag, left_gain = LC_flag, gain

        if left_LC_flag or right_LC_flag:
            if left_gain >= right_gain:
                left_prob, right_prob = MOBIL_LC_prob, 0.
            else:
                left_prob, right_prob = 0., MOBIL_LC_prob
            assert (left_prob + right_prob == 1)
        return left_prob, right_prob

    def check_whether_has_CF_data(self, f1):
        """
        Check whether there is car-following naturalistic data for the current step.
        If there is no naturalistic CF data, then use IDM+MOBIL.

        Args:
            f1: the leading vehicle.

        Returns:
            bool: True stands for there is data, False for not have data.
        """
        x, v = self.position[0], self.velocity
        r = f1.position[0] - x - self.LENGTH
        rr = f1.velocity - v
        round_speed, round_speed_idx = self.round_value_function(v, round_item="speed")
        round_r, round_r_idx = self.round_value_function(r, round_item="range")
        round_rr, round_rr_idx = self.round_value_function(rr, round_item="range_rate")

        pdf_array = global_val.CF_pdf_array[round_r_idx, round_rr_idx, round_speed_idx]
        if sum(pdf_array) == 0:
            return False
        else:
            return True

    def Longitudinal_NDD(self, obs, modify_flag=False):
        """
        Determine the longitudinal acceleration of the vehicle.

        Args:
            obs: vehicle observation.
            modify_flag: whether do real control. Since this function might be used to get vehicle naturalistic action probability.

        Returns:
            (float, ndarray): the sampled longitudinal acceleration and longitudinal acceleration probability mass function.
        """
        if not list(global_val.CF_pdf_array):
            assert ("No CF_pdf_array file!")
        if not list(global_val.FF_pdf_array):
            assert ("No FF_pdf_array file!")

        lane_id, x, v = self.lane_index[2], self.position[0], self.velocity
        f1, _, _, _, _, _ = obs

        if not f1:  # No vehicle ahead. Then FF
            round_speed, round_speed_idx = self.round_value_function(v, round_item="speed")
            pdf_array = global_val.FF_pdf_array[round_speed_idx]
            acc = np.random.choice(self.ACC_LIST, None, False, pdf_array)
            if modify_flag:
                self.mode, self.v, self.round_v, self.pdf_distribution = "FF", v, round_speed, pdf_array
            return acc, pdf_array

        else:  # Has vehicle ahead. Then CF
            r = f1.position[0] - x - self.LENGTH
            rr = f1.velocity - v
            round_speed, round_speed_idx = self.round_value_function(v, round_item="speed")
            round_r, round_r_idx = self.round_value_function(r, round_item="range")
            round_rr, round_rr_idx = self.round_value_function(rr, round_item="range_rate")

            if not self._check_bound_constraints(r, global_val.r_low, global_val.r_high) or not self._check_bound_constraints(rr, global_val.rr_low,
                                                                                                                              global_val.rr_high) or not \
                    self._check_bound_constraints(
                    v, global_val.v_low, global_val.v_high):  # Use stochastic IDM + MOBIL when no NDD.
                pdf_array = self.stochastic_IDM()
                if global_val.safety_guard_enabled_flag_IDM:
                    pdf_array = self._check_longitudinal_safety(obs, pdf_array)
                acc = np.random.choice(self.ACC_LIST, None, False, pdf_array)
                if modify_flag:
                    self.mode, self.pdf_distribution = "IDM", pdf_array
                    self.v, self.r, self.rr, self.round_v, self.round_r, self.round_rr, = v, r, rr, round_speed, round_r, round_rr
                return acc, pdf_array

            pdf_array = global_val.CF_pdf_array[round_r_idx, round_rr_idx, round_speed_idx]
            if sum(pdf_array) == 0:  # Use stochastic IDM + MOBIL when no NDD.
                pdf_array = self.stochastic_IDM()
                if global_val.safety_guard_enabled_flag_IDM:
                    pdf_array = self._check_longitudinal_safety(obs, pdf_array)
                acc = np.random.choice(self.ACC_LIST, None, False, pdf_array)
                if modify_flag:
                    self.mode, self.pdf_distribution = "IDM", pdf_array
                    self.v, self.r, self.rr, self.round_v, self.round_r, self.round_rr, = v, r, rr, round_speed, round_r, round_rr
                return acc, pdf_array
            acc = np.random.choice(self.ACC_LIST, None, False, pdf_array)
            if modify_flag:
                self.mode, self.pdf_distribution = "CF", pdf_array
                self.v, self.r, self.rr, self.round_v, self.round_r, self.round_rr, = v, r, rr, round_speed, round_r, round_rr
            return acc, pdf_array

    def stochastic_IDM(self):
        """
        Use a stochastic IDM model when there is no naturalistic driving data.

        Returns:
            list(float): probability mass function of longitudinal acceleration.
        """
        self.IDM_flag = True
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        tmp_acc = self.acceleration(ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle)
        tmp_acc = np.clip(tmp_acc, global_val.acc_low, global_val.acc_high)
        acc_possi_list = scipy.stats.norm.pdf(self.ACC_LIST, tmp_acc, 0.3)
        # Delete possi if smaller than certain threshold
        acc_possi_list = [val if val > global_val.Stochastic_IDM_threshold else 0 for val in acc_possi_list]
        assert (sum(acc_possi_list) > 0)
        acc_possi_list = acc_possi_list / (sum(acc_possi_list))

        return acc_possi_list
