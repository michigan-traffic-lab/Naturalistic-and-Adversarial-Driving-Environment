import numpy as np
import pandas as pd
import os
from highway_env import utils
from highway_env.envs.common.observation import KinematicObservation as observation

import torch
import torch.nn as nn
import global_val

torch.set_default_tensor_type(torch.FloatTensor)
device = "cpu"


class AV_RL_agent:
    """
    AV agent trained by deep reinforcement learning (DRL).
    """

    def __init__(self, env):
        self.observed_BV_num = env.cav_observation_num
        self.each_vehicle_feature = 3
        self.NUM_ACTION = len(global_val.ACTIONS)  # total number of actions
        self.feature_size = (self.observed_BV_num + 1) * self.each_vehicle_feature
        self.hidden_size = 256  # 150
        self.output_size = self.NUM_ACTION
        self.env = env

        # ===== Networks Initialization =====
        self.net = DuelingNet(self.feature_size, self.hidden_size, self.output_size).to(device)  # eval net

        # ====== Evaluation mode ======
        path = './Data/AV2_AGENT/100000_model_CAV_agent.pth'
        checkpoint = torch.load(path, map_location=device)
        self.net.load_state_dict(checkpoint['net'])
        self.net.eval()
        print("===================Load evaluation CAV model:", path)

    def decision(self, original_obs_df, action_indicator=None):
        """
        AV agent decision making.

        Args:
            original_obs_df: AV observation.
            action_indicator: safety indicator of each maneuver.

        Returns:
            int: action index.
        """

        '''
        Input original obs df, output action
        :param original_obs_df: original observation within the observation range
               action_indicator: [False, True,....] indicates each action whether safe or not, False for not safe, True for safe
        :return:
        '''
        # If all actions are unvoid, then decelerate
        if not action_indicator.any():
            action_indicator[2] = True
        assert action_indicator.any()

        state = self._transfer_to_state_input(original_obs_df)
        state = torch.from_numpy(state).float().to(device)
        action_Q_full = self.net(state).detach().cpu().numpy()
        action_Q_full[np.array(action_indicator) == 0] = -np.inf
        action_id = np.argmax(action_Q_full).item()

        return action_id

    def _transfer_to_state_input(self, original_obs_df):
        """
        Transfer observations to network state input.

        Args:
            original_obs_df: AV observation.

        Returns:
            network input.
        """
        # Normalize
        state_df = self._normalize_state(original_obs_df)
        # Fill with dummy vehicles
        if state_df.shape[0] < self.observed_BV_num + 1:
            fake_vehicle_row = [-1, -1, -1]  # at the back of observation box, with minimum speed and at the top lane
            fake_vehicle_rows = [fake_vehicle_row for _ in range(self.observed_BV_num + 1 - state_df.shape[0])]
            rows = np.array(fake_vehicle_rows)
            # rows = -np.ones((self.observed_BV_num + 1 - state_df.shape[0], len(observation.FEATURES_acc_training)))
            state_df = state_df.append(pd.DataFrame(data=rows, columns=observation.FEATURES_acc_training), ignore_index=True)

        return state_df.values.flatten()

    def _normalize_state(self, df):
        """
        Transfer the observation to relative states.

        Args:
            df: AV observation.

        Returns:
            normalized observations.
        """
        df_copy = df.copy()

        # Get relative BV data first
        df_copy.loc[1:, 'x'] = df_copy['x'][1:] - df_copy['x'][0]

        # Normalize BV
        x_position_range = self.env.cav_observation_range
        side_lanes = self.env.road.network.all_side_lanes(self.env.vehicle.lane_index)
        lane_num = len(side_lanes)
        lane_range = lane_num - 1

        df_copy.loc[1:, 'x'] = utils.remap(df_copy.loc[1:, 'x'], [-x_position_range, x_position_range], [-1, 1])
        df_copy.loc[1:, 'lane_id'] = utils.remap(df_copy.loc[1:, 'lane_id'], [0, lane_range], [-1, 1])
        df_copy.loc[1:, 'v'] = utils.remap(df_copy.loc[1:, 'v'], [self.env.min_velocity, self.env.max_velocity], [-1, 1])

        # Normalize CAV
        df_copy.loc[0, 'x'] = 0
        df_copy.loc[0, 'lane_id'] = utils.remap(df_copy.loc[0, 'lane_id'], [0, lane_range], [-1, 1])
        df_copy.loc[0, 'v'] = utils.remap(df_copy.loc[0, 'v'], [self.env.min_velocity, self.env.max_velocity], [-1, 1])

        assert ((-1.1 <= df_copy.x).all() and (df_copy.x <= 1.1).all())
        assert ((-1.1 <= df_copy.v).all() and (df_copy.v <= 1.1).all())
        assert ((-1.1 <= df_copy.lane_id).all() and (df_copy.lane_id <= 1.1).all())

        return df_copy

    def lane_conflict_safety_check(self, original_obs_df, action_indicator_before):
        """
        This function is to avoid potential lane-conflict dangerousness to improve the safety performance of the AV.

        Args:
            original_obs_df: AV observation.
            action_indicator_before: safety indicator of each maneuver before lane-conflict safety guard.

        Returns:
            safety indicator of each maneuver after lane-conflict safety guard.
        """
        # If there is no longitudinal actions are OK or in the middle lane, then do not block the lane change probability
        CAV_v, CAV_x, CAV_current_lane_id = self.env.vehicle.velocity, self.env.vehicle.position[0], self.env.vehicle.lane_index[2]
        if (not action_indicator_before[2:].any()) or (CAV_current_lane_id == 1):
            return action_indicator_before
        # If there is no lane change probability, just return
        if (CAV_current_lane_id == 0 and not action_indicator_before[1]) or (CAV_current_lane_id == 2 and not action_indicator_before[0]):
            return action_indicator_before

        if CAV_current_lane_id == 0:
            candidate_BV_lane, CAV_ban_lane_change_id = 2, 1
        elif CAV_current_lane_id == 2:
            candidate_BV_lane, CAV_ban_lane_change_id = 0, 0

        # candidate_BV
        candidate_BV_df = original_obs_df[original_obs_df["lane_id"] == candidate_BV_lane]
        if candidate_BV_df.shape[0] == 0:
            return action_indicator_before
        r_now, rr_now, r_1_second, r_2_second = [], [], [], []
        for row in candidate_BV_df.itertuples():
            BV_x, BV_v = row.x, row.v
            if BV_x >= CAV_x:
                r_now_tmp = BV_x - CAV_x - global_val.LENGTH
                rr_now_tmp = BV_v - CAV_v
                r_1_second_tmp = r_now_tmp + rr_now_tmp * global_val.simulation_resolution
                acc_BV = acc_CAV = global_val.acc_low
                BV_dis = utils.cal_dis_with_start_end_speed(BV_v, np.clip(BV_v + acc_BV, global_val.v_low, global_val.v_high), acc_BV,
                                                            time_interval=global_val.simulation_resolution)
                CAV_dis = utils.cal_dis_with_start_end_speed(CAV_v, np.clip(CAV_v + acc_CAV, global_val.v_low, global_val.v_high), acc_CAV,
                                                             time_interval=global_val.simulation_resolution)
                r_2_second_tmp = r_1_second_tmp + BV_dis - CAV_dis

                r_now.append(r_now_tmp)
                rr_now.append(rr_now_tmp)
                r_1_second.append(r_1_second_tmp)
                r_2_second.append(r_2_second_tmp)
            else:
                r_now_tmp = CAV_x - BV_x - global_val.LENGTH
                rr_now_tmp = CAV_v - BV_v
                r_1_second_tmp = r_now_tmp + rr_now_tmp * global_val.simulation_resolution
                acc_BV = global_val.acc_low
                acc_CAV = 0
                BV_dis = utils.cal_dis_with_start_end_speed(BV_v, np.clip(BV_v + acc_BV, global_val.v_low, global_val.v_high), acc_BV,
                                                            time_interval=global_val.simulation_resolution)
                CAV_dis = utils.cal_dis_with_start_end_speed(CAV_v, np.clip(CAV_v + acc_CAV, global_val.v_low, global_val.v_high), acc_CAV,
                                                             time_interval=global_val.simulation_resolution)
                r_2_second_tmp = r_1_second_tmp + CAV_dis - BV_dis

                r_now.append(r_now_tmp)
                rr_now.append(rr_now_tmp)
                r_1_second.append(r_1_second_tmp)
                r_2_second.append(r_2_second_tmp)

        r_now, r_1_second, r_2_second = np.array(r_now), np.array(r_1_second), np.array(r_2_second)
        if (r_now <= 0).any() or (r_1_second <= 0).any() or (r_2_second <= 0).any():
            # Sample to decide whether ban the lane change
            if np.random.rand() <= global_val.ignore_lane_conflict_prob:
                return action_indicator_before
            else:
                action_indicator_before[CAV_ban_lane_change_id] = False
                return action_indicator_before

        return action_indicator_before


class DuelingNet(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(DuelingNet, self).__init__()
        print("Using Dueling Network!")
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.adv = nn.Linear(n_hidden, n_output)
        self.val = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = torch.tanh(self.hidden(x))
        x = torch.tanh(self.hidden2(x))
        adv = self.adv(x)
        val = self.val(x)
        x = val + adv - adv.mean()
        return x
