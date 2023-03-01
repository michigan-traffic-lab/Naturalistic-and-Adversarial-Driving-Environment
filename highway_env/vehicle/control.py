from __future__ import division, print_function
import numpy as np
import copy
from highway_env import utils
from highway_env.vehicle.dynamics import Vehicle
import global_val
import uuid


class ControlledVehicle(Vehicle):
    """
        A vehicle piloted by two low-level controller, allowing high-level actions
        such as cruise control and lane changes.

        - The longitudinal controller is a velocity controller;
        - The lateral controller is a heading controller cascaded with a lateral position controller.
    """

    TAU_A = 0.6  # [s]
    TAU_DS = 0.2  # [s]
    PURSUIT_TAU = 1.5 * TAU_DS  # [s]
    KP_A = 1 / TAU_A
    KP_HEADING = 1 / TAU_DS
    KP_LATERAL = 1 / 0.5  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]

    DELTA_VELOCITY = 5  # [m/s]

    def __init__(self,
                 road,
                 position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=None,
                 route=None):
        super(ControlledVehicle, self).__init__(road, position, heading, velocity)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_velocity = target_velocity or self.velocity
        self.route = route
        self.actual_action = None
        self.longi_acc = 0
        self.id = str(uuid.uuid4())

        # Decomposed method
        self.weight = 1  # NDD probablity/Critical probability default is 1 for no manipulation
        self.criticality = 0  # Just used for BVs for those we controlled using decomposed method. For other vehicle such as CAV, this attribute has no meaning
        self.decomposed_controlled_flag = False  # Used for BV that has maximum criticality and selected to controll

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
                route=vehicle.route)
        return v

    def plan_route_to(self, destination):
        """
            Plan a route to a destination in the road network

        :param destination: a node in the road network
        """
        path = self.road.network.shortest_path(self.lane_index[1], destination)
        if path:
            self.route = [self.lane_index] + [(path[i], path[i + 1], None) for i in range(len(path) - 1)]
        else:
            self.route = [self.lane_index]
        return self

    def act(self, action=None, essential_flag=False):
        """
            Perform a high-level action to change the desired lane or velocity.

            - If a high-level action is provided, update the target velocity and lane;
            - then, perform longitudinal and lateral control.

        :param action: a high-level action
        """
        self.follow_road()
        _from, _to, _id = self.lane_index
        if essential_flag == 0:
            if action:
                self.actual_action = action
                if action == "LANE_RIGHT" and self.lane_index == len(self.road.network.graph[_from][_to]) - 1:
                    self.actual_action = "IDLE"
                elif action == "LANE_LEFT" and self.lane_index == 0:
                    self.actual_action = "IDLE"
            # if action == "FASTER":
            #     self.longi_acc = self.DELTA_VELOCITY

            # elif action == "SLOWER":
            #     self.longi_acc = -self.DELTA_VELOCITY
            if action == "LANE_RIGHT":
                self.longi_acc = 0
                _from, _to, _id = self.target_lane_index
                target_lane_index = _from, _to, np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1)
                if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                    self.target_lane_index = target_lane_index
            elif action == "LANE_LEFT":
                self.longi_acc = 0
                _from, _to, _id = self.target_lane_index
                target_lane_index = _from, _to, np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1)
                if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                    self.target_lane_index = target_lane_index
            elif action:
                self.longi_acc = float(action)

        action = {'steering': self.steering_control(self.target_lane_index),
                  'acceleration': self.longi_acc}
        super(ControlledVehicle, self).act(action)

    def follow_road(self):
        """
           At the end of a lane, automatically switch to a next one.
        """
        if self.road.network.get_lane(self.target_lane_index).after_end(self.position):
            self.target_lane_index = self.road.network.next_lane(self.target_lane_index, route=self.route, position=self.position, np_random=self.road.np_random)
            # print(self.target_lane_index)

    def steering_control(self, target_lane_index):
        """
            Steer the vehicle to follow the center of an given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral velocity command
        2. Lateral velocity command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        target_lane = self.road.network.get_lane(target_lane_index)
        lane_coords = target_lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.velocity * self.PURSUIT_TAU
        lane_future_heading = target_lane.heading_at(lane_next_coords)

        # Lateral position control
        lateral_velocity_command = - self.KP_LATERAL * lane_coords[1]
        # Lateral velocity to heading
        heading_command = np.arcsin(np.clip(lateral_velocity_command / utils.not_zero(self.velocity), -1, 1))
        heading_ref = lane_future_heading + np.clip(heading_command, -np.pi / 4, np.pi / 4)
        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(heading_ref - self.heading)
        # Heading rate to steering angle
        steering_angle = self.LENGTH / utils.not_zero(self.velocity) * np.arctan(heading_rate_command)
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        return steering_angle

    def velocity_control(self, target_velocity):
        """
            Control the velocity of the vehicle.

            Using a simple proportional controller.

        :param target_velocity: the desired velocity
        :return: an acceleration command [m/s2]
        """
        return self.KP_A * (target_velocity - self.velocity)

    def set_route_at_intersection(self, _to):
        """
            Set the road to be followed at the next intersection.
            Erase current planned route.
        :param _to: index of the road to follow at next intersection, in the road network
        """

        if not self.route:
            return
        for index in range(min(len(self.route), 3)):
            try:
                next_destinations = self.road.network.graph[self.route[index][1]]
            except KeyError:
                continue
            if len(next_destinations) >= 2:
                break
        else:
            return
        next_destinations_from = list(next_destinations.keys())
        if _to == "random":
            _to = self.road.np_random.randint(0, len(next_destinations_from))
        next_index = _to % len(next_destinations_from)
        self.route = self.route[0:index + 1] + \
                     [(self.route[index][1], next_destinations_from[next_index], self.route[index][2])]

    def get_action_indicator(self, ndd_flag=False, safety_flag=True, CAV_flag=False):
        """
        Get AV action indicator.

        Args:
            ndd_flag: whether bound by naturalistic data.
            safety_flag: whether bound by safety.
            CAV_flag: whether is AV.

        Returns:
            np.array with the same size of the AV action.
        """
        if CAV_flag:
            action_shape = len(global_val.ACTIONS)
            ndd_action_indicator = np.ones(action_shape)
            if ndd_flag:
                pass
            safety_action_indicator = np.ones(action_shape)
            if safety_flag:
                obs = self._get_veh_obs()
                lateral_action_indicator = np.array([1, 1, 1])
                lateral_result = self._check_lateral_safety(obs, lateral_action_indicator, CAV_flag=True)
                longi_result = self._check_longitudinal_safety(obs, np.ones(action_shape - 2), lateral_result=lateral_result, CAV_flag=True)
                safety_action_indicator[0], safety_action_indicator[1] = lateral_result[0], lateral_result[2]
                safety_action_indicator[2:] = longi_result
            action_indicator = ndd_action_indicator * safety_action_indicator
            action_indicator = (action_indicator > 0)
            return action_indicator
        else:
            raise ValueError("Get BV action Indicator in CAV function")

    def _check_longitudinal_safety(self, obs, pdf_array, lateral_result=None, CAV_flag=False):
        """
        Model-based safety guard for each longitudinal acceleration.

        Args:
            obs: observation.
            pdf_array: potential action array.
            lateral_result: lateral feasibility
            CAV_flag: whether current checking vehicle is AV.

        Returns:
            whether each maneuver is safe (>0 or True for safe, =0 or False for unsafe).
        """
        f_veh, _, _, _, _, _ = obs
        safety_buffer = global_val.longi_safety_buffer
        for i in range(len(pdf_array) - 1, -1, -1):
            if CAV_flag:
                acc = global_val.CAV_acc_to_idx_dic.inverse[i]
            else:
                acc = global_val.acc_to_idx_dic.inverse[i]
            if f_veh:
                rr = f_veh.velocity - self.velocity
                r = f_veh.position[0] - self.position[0] - self.LENGTH
                criterion_1 = rr * global_val.simulation_resolution + r + 0.5 * (global_val.acc_low - acc) * global_val.simulation_resolution ** 2
                self_v_2, f_v_2 = max(self.velocity + acc, global_val.v_low), max((f_veh.velocity + global_val.acc_low), global_val.v_low)
                dist_r = (self_v_2 ** 2 - global_val.v_low ** 2) / (2 * abs(global_val.acc_low))
                dist_f = (f_v_2 ** 2 - global_val.v_low ** 2) / (2 * abs(global_val.acc_low)) + global_val.v_low * (f_v_2 - self_v_2) / global_val.acc_low
                criterion_2 = criterion_1 - dist_r + dist_f
                if criterion_1 <= safety_buffer or criterion_2 <= safety_buffer:
                    pdf_array[i] = 0
                    # if CAV_flag:
                    #     print(pdf_array)
                else:
                    break

        # Only set the decelerate most when non of lateral is OK.
        if lateral_result is not None:
            lateral_feasible = lateral_result[0] or lateral_result[2]
        else:
            lateral_feasible = False
        if np.sum(pdf_array) == 0 and not lateral_feasible:
            pdf_array[0] = 1 if not CAV_flag else np.exp(-2)
            return pdf_array

        if CAV_flag:
            new_pdf_array = pdf_array
        else:
            new_pdf_array = pdf_array / np.sum(pdf_array)
        return new_pdf_array

    def _check_lateral_safety(self, obs, pdf_array, CAV_flag=False):
        """
        Model-based safety guard for each lateral maneuver.

        Args:
            obs: observation.
            pdf_array: potential action array.
            CAV_flag: whether current checking vehicle is AV.

        Returns:
            whether each maneuver is safe (>0 or True for safe, =0 or False for unsafe). [Left turn, go straight, right turn],

        """
        f1, r1, f0, r0, f2, r2 = obs
        lane_change_dir = [0, 2]
        nearby_vehs = [[f0, r0], [f2, r2]]
        safety_buffer = global_val.lateral_safety_buffer
        if self.lane_index[2] == 0:
            pdf_array[0] = 0
        elif self.lane_index[2] == 2:
            pdf_array[2] = 0
        for lane_index, nearby_veh in zip(lane_change_dir, nearby_vehs):
            if pdf_array[lane_index] != 0:
                f_veh, r_veh = nearby_veh[0], nearby_veh[1]
                if f_veh:
                    rr = f_veh.velocity - self.velocity
                    r = f_veh.position[0] - self.position[0] - self.LENGTH
                    dis_change = rr * global_val.simulation_resolution + 0.5 * global_val.acc_low * global_val.simulation_resolution ** 2
                    r_1 = r + dis_change
                    rr_1 = rr + global_val.acc_low * global_val.simulation_resolution

                    if r_1 <= safety_buffer or r <= safety_buffer:
                        pdf_array[lane_index] = 0
                    elif rr_1 < 0:
                        self_v_2, f_v_2 = max(self.velocity, global_val.v_low), max((f_veh.velocity + global_val.acc_low), global_val.v_low)
                        dist_r = (self_v_2 ** 2 - global_val.v_low ** 2) / (2 * abs(global_val.acc_low))
                        dist_f = (f_v_2 ** 2 - global_val.v_low ** 2) / (2 * abs(global_val.acc_low)) + global_val.v_low * (f_v_2 - self_v_2) / global_val.acc_low
                        r_2 = r_1 - dist_r + dist_f
                        if r_2 <= safety_buffer:
                            pdf_array[lane_index] = 0

                if r_veh:
                    rr = self.velocity - r_veh.velocity
                    r = self.position[0] - r_veh.position[0] - self.LENGTH
                    dis_change = rr * 1 - 0.5 * global_val.acc_high * 1 ** 2
                    r_1 = r + dis_change
                    rr_1 = rr - global_val.acc_high * 1
                    if r_1 <= safety_buffer or r <= safety_buffer:
                        pdf_array[lane_index] = 0
                    elif rr_1 < 0:
                        self_v_2, r_v_2 = min(self.velocity, global_val.v_high), min((r_veh.velocity + global_val.acc_high), global_val.v_high)
                        dist_r = (r_v_2 ** 2 - global_val.v_low ** 2) / (2 * abs(global_val.acc_low))
                        dist_f = (self_v_2 ** 2 - global_val.v_low ** 2) / (2 * abs(global_val.acc_low)) + global_val.v_low * (-r_v_2 + self_v_2) / global_val.acc_low
                        r_2 = r_1 - dist_r + dist_f
                        if r_2 <= safety_buffer:
                            pdf_array[lane_index] = 0
        if np.sum(pdf_array) == 0:
            return np.array([0, 1, 0])

        if CAV_flag:
            new_pdf_array = pdf_array
        else:
            new_pdf_array = pdf_array / np.sum(pdf_array)
        return new_pdf_array

    def _get_veh_obs(self):
        """
        Get vehicle surround observations. f0, f1, f2 denote closest other vehicles in front on the subject vehicle in the left adjacent lane, same lane, right adjacent lane.
        r0, r1, r2 denote the corresponding ones behind the subject vehicle.

        Returns:
            observations.
        """
        lane_id = self.lane_index[2]
        observation = []  # observation for this vehicle
        if lane_id == 0:
            f0, r0 = None, None
            f1, r1 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 0), obs_range=global_val.cav_obs_range)
            f2, r2 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 1), obs_range=global_val.cav_obs_range)
            observation = [f1, r1, f0, r0, f2, r2]
        if lane_id == 1:
            f0, r0 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 0), obs_range=global_val.cav_obs_range)
            f1, r1 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 1), obs_range=global_val.cav_obs_range)
            f2, r2 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 2), obs_range=global_val.cav_obs_range)
            observation = [f1, r1, f0, r0, f2, r2]
        if lane_id == 2:
            f0, r0 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 1), obs_range=global_val.cav_obs_range)
            f1, r1 = self.road.neighbour_vehicles_within_range(self, ("a", "b", 2), obs_range=global_val.cav_obs_range)
            f2, r2 = None, None
            observation = [f1, r1, f0, r0, f2, r2]
        return observation


class MDPVehicle(ControlledVehicle):
    """
        A controlled vehicle with a specified discrete range of allowed target velocities.
    """

    SPEED_COUNT = 3  # []
    SPEED_MIN = 20  # [m/s]
    SPEED_MAX = 30  # [m/s]

    # CAV surrogate model Longitudinal policy parameters
    COMFORT_ACC_MAX = global_val.SM_IDM_COMFORT_ACC_MAX  # [m/s2]  2
    COMFORT_ACC_MIN = global_val.SM_IDM_COMFORT_ACC_MIN  # [m/s2]  -4
    DISTANCE_WANTED = global_val.SM_IDM_DISTANCE_WANTED  # [m]  5
    TIME_WANTED = global_val.SM_IDM_TIME_WANTED  # [s]  1.5
    DESIRED_VELOCITY = global_val.SM_IDM_DESIRED_VELOCITY  # [m/s]
    DELTA = global_val.SM_IDM_DELTA  # []

    def __init__(self,
                 road,
                 position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=None,
                 route=None):
        super(MDPVehicle, self).__init__(road, position, heading, velocity, target_lane_index, target_velocity, route)
        # self.velocity_index = self.speed_to_index(self.target_velocity)
        # self.target_velocity = self.index_to_speed(self.velocity_index)

    def acceleration(self, ego_vehicle, front_vehicle=None, rear_vehicle=None):
        """
            Compute an acceleration command with the Intelligent Driver Model.

            The acceleration is chosen so as to:
            - reach a target velocity;
            - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
                            reason about other vehicles behaviors even though they may not IDMs.
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

    def act(self, action=None, essential_flag=False):
        """
            Perform a high-level action.

            If the action is a velocity change, choose velocity from the allowed discrete range.
            Else, forward action to the ControlledVehicle handler.

        :param action: a high-level action
        """
        super(MDPVehicle, self).act(action, essential_flag=essential_flag)
        # if action == "FASTER":
        #     self.velocity_index = self.speed_to_index(self.velocity) + 1
        # elif action == "SLOWER":
        #     self.velocity_index = self.speed_to_index(self.velocity) - 1
        # else:
        #     super(MDPVehicle, self).act(action)
        #     return
        # self.velocity_index = np.clip(self.velocity_index, 0, self.SPEED_COUNT - 1)
        # self.target_velocity = self.index_to_speed(self.velocity_index)
        # super(MDPVehicle, self).act()

    @classmethod
    def index_to_speed(cls, index):
        """
            Convert an index among allowed speeds to its corresponding speed
        :param index: the speed index []
        :return: the corresponding speed [m/s]
        """
        if cls.SPEED_COUNT > 1:
            return cls.SPEED_MIN + index * (cls.SPEED_MAX - cls.SPEED_MIN) / (cls.SPEED_COUNT - 1)
        else:
            return cls.SPEED_MIN

    @classmethod
    def speed_to_index(cls, speed):
        """
            Find the index of the closest speed allowed to a given speed.
        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - cls.SPEED_MIN) / (cls.SPEED_MAX - cls.SPEED_MIN)
        return np.int(np.clip(np.round(x * (cls.SPEED_COUNT - 1)), 0, cls.SPEED_COUNT - 1))

    def speed_index(self):
        """
            The index of current velocity
        """
        return self.speed_to_index(self.velocity)

    def predict_trajectory(self, actions, action_duration, trajectory_timestep, dt):
        """
            Predict the future trajectory of the vehicle given a sequence of actions.

        :param actions: a sequence of future actions.
        :param action_duration: the duration of each action.
        :param trajectory_timestep: the duration between each save of the vehicle state.
        :param dt: the timestep of the simulation
        :return: the sequence of future states
        """
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.act(action)  # High-level decision
            for _ in range(int(action_duration / dt)):
                t += 1
                v.act()  # Low-level control action
                v.step(dt)
                if (t % int(trajectory_timestep / dt)) == 0:
                    states.append(copy.deepcopy(v))
        return states
