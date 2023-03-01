from __future__ import division, print_function, absolute_import
from gym.envs.registration import register
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.control import ControlledVehicle, MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle
import global_val
import numpy as np


class HighwayExitEnv(AbstractEnv):
    """
        A highway merge negotiation environment.

        The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
        It is rewarded for maintaining a high velocity and avoiding collisions, but also making room for merging
        vehicles.
    """
    COLLISION_REWARD = -1
    RIGHT_LANE_REWARD = 0.1
    HIGH_VELOCITY_REWARD = 0.2
    MERGING_VELOCITY_REWARD = -0.5
    LANE_CHANGE_REWARD = -0.05
    EXIT_LENGTH = global_val.EXIT_LENGTH
    HIGHWAY_LENGTH = global_val.HIGHWAY_LENGTH

    DEFAULT_CONFIG = {
        "observation": {
            "type": "Kinematics"
        },
        "policy_frequency": 1,  # [Hz]
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "screen_width": 600,
        "screen_height": 150,
        "centering_position": [0.3, 0.5]
    }

    def __init__(self):
        self.mode = None
        super(HighwayExitEnv, self).__init__()

    def _reward(self, action):
        """
            The vehicle is rewarded for driving with high velocity on lanes to the right and avoiding collisions, but
            an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low velocity.
        :param action: the action performed
        :return: the reward of the state-action transition
        """

        # Tmp Add to
        return 0
        # action_reward = {0: self.LANE_CHANGE_REWARD, 1: 0, 2: self.LANE_CHANGE_REWARD, 3: 0, 4: 0}
        reward = self.COLLISION_REWARD * self.vehicle.crashed + self.RIGHT_LANE_REWARD * \
                 self.vehicle.lane_index[2] / 1 + self.HIGH_VELOCITY_REWARD * self.vehicle.velocity_index / \
                 (self.vehicle.SPEED_COUNT - 1)

        # Altruistic penalty
        for vehicle in self.road.vehicles:
            if vehicle.lane_index == ("b", "c", 2) and isinstance(vehicle, ControlledVehicle):
                reward += self.MERGING_VELOCITY_REWARD * \
                          (vehicle.target_velocity - vehicle.velocity) / vehicle.target_velocity

        return utils.remap(action_reward[action] + reward,
                           [self.COLLISION_REWARD, self.HIGH_VELOCITY_REWARD + self.RIGHT_LANE_REWARD],
                           [0, 1])

    def _is_terminal(self):
        """
            The episode is over when a collision occurs or when the access ramp has been passed.
        """
        _from, _to, _id = self.vehicle.lane_index
        last_id = len(self.vehicle.road.network.graph[_from][_to]) - 1
        exit_flag = self.vehicle.position[0] < self.EXIT_LENGTH and _id == last_id
        return self.vehicle.crashed or self.vehicle.position[0] > self.HIGHWAY_LENGTH or exit_flag

    def reset(self):
        # self._make_road()
        # self._make_vehicles()
        return super(HighwayExitEnv, self).reset()

    def _make_road(self):
        """
            Make a road composed of a straight highway and a merging lane.
        :return: the road
        """
        net = RoadNetwork()
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH, 2 * StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, s], [n, c]]
        for i in range(3):
            net.add_lane("a", "b", StraightLane([0, y[i]], [self.HIGHWAY_LENGTH, y[i]], line_types=line_type[i]))
        road = Road(network=net, np_random=self.np_random)
        self.road = road

    def _make_vehicles(self, background_vehicle=None,
                       auto_vehicle=(0, 50, 24)):
        """
            Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        background_vehicle: Each item has [lane,position,velocity]
        """
        road = self.road
        ego_vehicle = MDPVehicle(road, road.network.get_lane(("a", "b", auto_vehicle[0])).position(auto_vehicle[1], 0),
                                 velocity=auto_vehicle[2])
        road.vehicles.append(ego_vehicle)
        if self.mode == "ACC":
            other_vehicles_type = MDPVehicle
        elif self.mode == "NDD":
            other_vehicles_type = IDMVehicle
        else:
            raise NotImplementedError("Not supported mode")
        for lane_id in range(3):
            if lane_id != 0:
                position_tmp = np.random.uniform(low=0, high=100)
                velocity_tmp = 30
                road.vehicles.append(
                    other_vehicles_type(road, road.network.get_lane(("a", "b", lane_id)).position(position_tmp, 0),
                                        velocity=velocity_tmp))
            else:
                position_tmp = auto_vehicle[1]
                velocity_tmp = auto_vehicle[2]
            position_tmp = position_tmp + 50
            velocity_tmp = 20
            road.vehicles.append(
                other_vehicles_type(road, road.network.get_lane(("a", "b", lane_id)).position(position_tmp, 0),
                                    velocity=velocity_tmp))
        self.vehicle = ego_vehicle

    def step(self, action):
        """
            Perform an action and step the environment dynamics.

            The action is executed by the ego-vehicle, and all other vehicles on the road performs their default
            behaviour for several simulation timesteps until the next decision making step.
        :param int action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminal, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self._simulate(action)

        obs = self.observation.observe()
        reward = self._reward(action)
        terminal = self._is_terminal()

        # cost = self._cost(action)
        # SHW add 20190728 exit_Flag in info
        if not terminal:
            info = None
        else:
            _from, _to, _id = self.vehicle.lane_index
            last_id = len(self.vehicle.road.network.graph[_from][_to]) - 1
            exit_flag = self.vehicle.position[0] > self.EXIT_LENGTH and _id == last_id
            info = exit_flag

        return obs, reward, terminal, info


register(
    id='highway-exit-v0',
    entry_point='highway_env.envs:HighwayExitEnv',
)
