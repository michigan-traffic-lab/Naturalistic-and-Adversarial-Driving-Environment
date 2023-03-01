import global_val
from highway_env import utils
import numpy as np
import yaml

from highway_env.vehicle.behavior import NDDVehicle  # Background vehicle

# ============== Algorithm parameters ============
# Load config file
with open('config/configs.yml') as file:
    try:
        configs = yaml.safe_load(file)
    except yaml.YAMLError as exception:
        print(exception)

simulation_resolution = global_val.simulation_resolution  # The time resolution of the simulation platform (second)
epsilon_value = configs['epsilon_value']
BV_model = NDDVehicle
num_controlled_critical_bvs = 1
criticality_threshold = 0


def select_controlled_bv_and_action(obs, env):
    """
    This function:

        - calculates the criticality of each candidate controlled BV.
        - sample the action according to importance function if the criticality greater than the threshold.
        - select the BV with the highest criticality to control.

    Args:
        obs: AV observation of its surrounding vehicles.
        env: simulation environment.

    Returns:
        (list, list, float, list, list):
        - bv_action_idx_list - the action idx list for each candidate controlled BV. For BV with the maximum criticality, the action idx is sampled from the importance function. For other BVs, the action idx is None.
        - weight_list - the list of weight (importance sampling likelihood) for each candidate controlled BV.
        - max_criticality - the max criticality at this moment.
        - ndd_possi_list - the list of the sampled action naturalistic probability.
        - critical_possi_list - the list of the sampled action importance function probability.

    """
    CAV_left_prob, CAV_still_prob, CAV_right_prob = _get_Surrogate_CAV_action_probability(env)  # Use surrogate model to predict CAV behavior.
    # Initialize the criticality/action/weight/ndd_probability/importance_function_probability list correspond to each candidate controlled BV.
    bv_criticality_list, bv_action_idx_list, weight_list, ndd_possi_list, critical_possi_list = [], [], [], [], []
    for i in range(len(env.controlled_bvs)):  # Loop over all candidate controlled BVs.
        bv, obs_bv = env.controlled_bvs[i], obs[i]
        # Calculate criticality, action, ..., for specific BV.
        bv_criticality, bv_action_idx, weight, ndd_possi, critical_possi = Decompose_decision(bv, obs_bv, AV_SM_prob=(CAV_left_prob, CAV_still_prob, CAV_right_prob), env=env)
        if bv_action_idx is not None:  # If criticality > threshold, then record this BV sampled action.
            bv_action_idx = bv_action_idx.item()
            bv.criticality = bv_criticality  # save the criticality for the candidate BV.
        bv_criticality_list.append(bv_criticality), bv_action_idx_list.append(bv_action_idx), weight_list.append(weight), ndd_possi_list.append(
            ndd_possi), critical_possi_list.append(critical_possi)
    tmp_check = np.array([True if ((bv_criticality_list[i] > criticality_threshold) and bv_action_idx_list[i] is not None) or (
            (bv_criticality_list[i] <= criticality_threshold) and not bv_action_idx_list[i]) else False for i in range(len(env.controlled_bvs))])
    assert (tmp_check.all())

    # Select the BV with highest criticality to control.
    selected_bv_idx = sorted(range(len(bv_criticality_list)), key=lambda x: bv_criticality_list[x])[-num_controlled_critical_bvs:]
    for i in range(len(env.controlled_bvs)):
        if i not in selected_bv_idx:
            bv_action_idx_list[i], weight_list[i], ndd_possi_list[i], critical_possi_list[i] = None, None, None, None
        else:
            if weight_list[i]:  # If the weight is not None and also greater than 0.
                decomposed_controlled_bv = env.controlled_bvs[i]
                decomposed_controlled_bv.weight = weight_list[i]
                decomposed_controlled_bv.decomposed_controlled_flag = True

    if len(bv_criticality_list):
        max_criticality = np.max(bv_criticality_list)  # max criticality at this moment.
    else:
        max_criticality = -np.inf

    return bv_action_idx_list, weight_list, max_criticality, ndd_possi_list, critical_possi_list


def epsilon_greedy(pdf_before_epsilon, ndd_pdf, epsilon=0.1):
    """
    The epsilon is distributed to all actions according to the exposure frequency.

    Args:
        pdf_before_epsilon: importance function action probability distribution before epsilon greedy.
        ndd_pdf: naturalistic action probability distribution.
        epsilon: the epsilon value.

    Returns:
        list(float): importance function action probability distribution after epsilon greedy.
    """
    # NDD epsilon greedy method
    pdf_after_epsilon = (1 - epsilon) * pdf_before_epsilon + epsilon * ndd_pdf
    assert (0.99999 <= np.sum(pdf_after_epsilon) <= 1.0001)
    return pdf_after_epsilon


def get_NDD_possi(bv, obs_bv):
    """
    Obtain the BV naturalistic action probability distribution.

    Args:
        bv: the specific BV.
        obs_bv: BV's observation.

    Returns:
        list(float): the probability mass function for different actions. The first element is the left lane change probability, the second element is the right lane change probability, the followed by
        longitudinal acceleration probability.

    """

    possi_array = np.zeros((len(global_val.BV_ACTIONS)), dtype=float)
    _, _, lat_possi_array = bv.Lateral_NDD(obs_bv, modify_flag=False)
    _, long_possi_array = bv.Longitudinal_NDD(obs_bv, modify_flag=False)
    possi_array[0], possi_array[1] = lat_possi_array[0], lat_possi_array[2]
    possi_array[2:] = lat_possi_array[1] * long_possi_array
    return possi_array


def _round_data(bv, v, r, rr):
    """
    Round the continuous state to vehicle's decision state resolution. For example, the current velocity is 30.18, if the resolution is 1 then round to 30.

    Args:
        bv: the specific BV.
        v: velocity.
        r: range (spacing).
        rr: range rate (relative speed). Front vehicle speed - following vehicle speed.

    Returns:
        rounded value.
    """

    round_speed, round_speed_idx = bv.round_value_function(v, round_item="speed")
    round_r, round_r_idx = bv.round_value_function(r, round_item="range")
    round_rr, round_rr_idx = bv.round_value_function(rr, round_item="range_rate")

    return round_speed, round_speed_idx, round_r, round_r_idx, round_rr, round_rr_idx


def _sample_critical_action(criticality_array, bv_criticality, possi_array):
    """
    Construct importance function and then sample BV action according to it with epsilon-greedy.

    Args:
        criticality_array: criticality values for all BV actions.
        bv_criticality: the criticality of the BV (sum of the criticality array).
        possi_array: naturalistic probability for all BV actions.

    Returns:
        (int, float, float, float):
        - bv_action_idx - sampled BV action index.
        - weight - importance sampling likelihood.
        - ndd_possi - naturalistic probability of the sampled action.
        - critical_possi - importance function probability of the sampled action.
    """
    pdf_array = criticality_array / bv_criticality
    pdf_array = epsilon_greedy(pdf_array, possi_array, epsilon=epsilon_value)
    bv_action_idx = np.random.choice(len(global_val.BV_ACTIONS), 1, replace=False, p=pdf_array)
    critical_possi, ndd_possi = pdf_array[bv_action_idx], possi_array[bv_action_idx]
    weight = ndd_possi / critical_possi
    return bv_action_idx, weight, ndd_possi, critical_possi


def _lane_change_with_CAV_challenge(bv, v, r, rr, obs_bv, action_idx, env):
    """
    Calculate the lane change challenge value when the BV is in front of the CAV, and then do the same LC with the CAV. It is composed by two parts
    - challenge during the LC (crash happens during the LC process).
    - challenge after the LC (crash happens in the car-following situation after the LC process).

    Args:
        bv: the BV.
        v: velocity of the BV.
        r: the range (spacing) between BV and the following AV.
        rr: the range rate (relative speed) between BV and the following AV.
        obs_bv: BV observation.
        action_idx: 0 or 1. 0 stands for left turn, 1 stands for right turn.
        env: the simulation environment.

    Returns:
        float: the challenge value of this situation.
    """
    challenge = 0

    _, _, f0, _, f2, _ = obs_bv
    v_after_LC, rr_after_LC = v, rr
    r_after_LC = r + rr * simulation_resolution

    if r_after_LC <= 0:  # Crash during the LC.
        challenge = 1
        return challenge
    else:  # The challenge in car-following situation after LC.
        round_speed_after_LC, _, round_r_after_LC, _, round_rr_after_LC, _ = _round_data(bv, v_after_LC, r_after_LC, rr_after_LC)

        index = np.where((global_val.CF_state_value == [round_r_after_LC, round_rr_after_LC, round_speed_after_LC]).all(1))
        assert (len(index) <= 1)
        index = index[0]

        if len(index):
            CF_challenge_array = global_val.CF_challenge_value[index.item(), :]  # Car-following challenge learned off-line using RL.

            if action_idx == 0:
                front_veh_after_LC = f0  # The new leading vehicle for the BV after it LC to left.
            elif action_idx == 1:
                front_veh_after_LC = f2  # The new leading vehicle for the BV after it LC to right.

            if front_veh_after_LC:
                predict_v, predict_x, predict_lane_id = front_veh_after_LC.velocity, front_veh_after_LC.position[0] + front_veh_after_LC.velocity * simulation_resolution, \
                                                        front_veh_after_LC.lane_index[2]  # Predicted states for BV's new leading vehicle.
                predict_front_veh_after_LC = BV_model(env.road, env.road.network.get_lane(("a", "b", predict_lane_id)).position(predict_x, 0), 0, predict_v)
                predict_self_v, predict_self_x, predict_self_lane_id = bv.velocity, bv.position[0] + bv.velocity * simulation_resolution, front_veh_after_LC.lane_index[2]
                predict_self_after_LC = BV_model(env.road, env.road.network.get_lane(("a", "b", predict_self_lane_id)).position(predict_self_x, 0), 0, predict_self_v)
            else:  # The BV is in free-flow situation if there is no vehicle in front of it after LC.
                predict_front_veh_after_LC = None
                predict_self_after_LC = bv

            obs_bv_after_LC = [predict_front_veh_after_LC, None, None, None, None, None]
            _, long_possi_after_LC_array = predict_self_after_LC.Longitudinal_NDD(obs_bv_after_LC, modify_flag=False)  # BV acceleration probability distribution after LC.
            challenge = np.sum(long_possi_after_LC_array * CF_challenge_array)  # challenge consider all possible acceleration manuevers.

    try:
        assert (challenge <= 1.00001)
    except:
        print(challenge)
        print(long_possi_after_LC_array)
        print(CF_challenge_array)
        raise ValueError("Challenge")
    return challenge


def _BV_behind_lane_change_with_CAV_challenge(bv, v, r, rr, action_idx, env):
    """
    Calculate the lane change challenge value when the BV is behind the CAV, and then do the same LC with the CAV.

    Args:
        bv: the BV.
        v: velocity of the BV.
        r: the range (spacing) between BV and the following AV.
        rr: the range rate (relative speed) between BV and the following AV.
        action_idx: 0 or 1. 0 stands for left turn, 1 stands for right turn.
        env: the simulation environment.

    Returns:
        float: the challenge value of this situation.

    """
    challenge = 0

    cav = env.vehicle
    v_after_LC, rr_after_LC = v, rr
    r_after_LC = r + rr * simulation_resolution

    if r_after_LC <= 0:
        challenge = 1
        return challenge
    else:
        round_speed_after_LC, _, round_r_after_LC, _, round_rr_after_LC, _ = _round_data(bv, v_after_LC, r_after_LC, rr_after_LC)

        index = np.where((global_val.BV_CF_state_value == [round_r_after_LC, round_rr_after_LC, round_speed_after_LC]).all(1))
        assert (len(index) <= 1)
        index = index[0]

        if len(index):
            BV_CF_challenge_array = global_val.BV_CF_challenge_value[index.item(), :]

            if action_idx == 0:
                predict_cav_lane_id = bv.lane_index[2] - 1
            elif action_idx == 1:
                predict_cav_lane_id = bv.lane_index[2] + 1

            predict_cav_v = cav.velocity
            predict_cav_x = cav.position[0] + cav.velocity * simulation_resolution,
            predict_front_veh_after_LC = BV_model(env.road, env.road.network.get_lane(("a", "b", predict_cav_lane_id)).position(predict_cav_x, 0), 0, predict_cav_v)

            predict_self_v, predict_self_x, predict_self_lane_id = bv.velocity, bv.position[0] + bv.velocity * simulation_resolution, predict_cav_lane_id
            predict_self_after_LC = BV_model(env.road, env.road.network.get_lane(("a", "b", predict_self_lane_id)).position(predict_self_x, 0), 0, predict_self_v)

            obs_bv_after_LC = [predict_front_veh_after_LC, None, None, None, None, None]
            _, long_possi_after_LC_array = predict_self_after_LC.Longitudinal_NDD(obs_bv_after_LC, modify_flag=False)
            challenge = np.sum(long_possi_after_LC_array * BV_CF_challenge_array)

    try:
        assert (challenge <= 1.00001)
    except:
        print(challenge)
        print(long_possi_after_LC_array)
        print(BV_CF_challenge_array)
        raise ValueError("_BV_behind_lane_change_with_CAV_challenge Challenge")
    return challenge


def _hard_brake_challenge(bv, v, r, rr):
    """
    Calculate the BV all longitudinal acceleration maneuvers (hard brake) challenge value when the BV is in front of the CAV.

    Args:
        bv: the BV.
        v: velocity of the BV.
        r: the range (spacing) between BV and the following AV.
        rr: the range rate (relative speed) between BV and the following AV.

    Returns:
        float: the challenge value of this situation.

    """
    CF_challenge_array = np.zeros((len(global_val.BV_ACTIONS) - 2), dtype=float)
    round_speed, _, round_r, _, round_rr, _ = _round_data(bv, v, r, rr)

    index = np.where((global_val.CF_state_value == [round_r, round_rr, round_speed]).all(1))
    assert (len(index) <= 1)
    index = index[0]

    if len(index):
        CF_challenge_array = global_val.CF_challenge_value[index.item(), :]
    return CF_challenge_array


def _cut_in_challenge(bv, v, r, rr, obs_bv, action_idx, env):
    """
    Calculate the BV lane change (cut-in to AV's lane) challenge value. It is composed by two parts
    - challenge during the LC (crash happens during the LC process).
    - challenge after the LC (crash happens in the car-following situation after the LC process).

    Args:
        bv: the BV.
        v: velocity of the BV.
        r: the range (spacing) between BV and the following AV.
        rr: the range rate (relative speed) between BV and the following AV.
        obs_bv: BV observation.
        action_idx: 0 or 1. 0 stands for left turn, 1 stands for right turn.
        env: the simulation environment.

    Returns:
        float: the challenge value of this situation.

    """

    challenge = 0

    _, _, f0, _, f2, _ = obs_bv
    v_after_LC, rr_after_LC = v, rr
    r_after_LC = r + rr * simulation_resolution

    if r_after_LC <= 0:  # Crash during the LC.
        challenge = 1
    else:  # The challenge in car-following situation after LC.
        round_speed_after_LC, _, round_r_after_LC, _, round_rr_after_LC, _ = _round_data(bv, v_after_LC, r_after_LC, rr_after_LC)

        index = np.where((global_val.CF_state_value == [round_r_after_LC, round_rr_after_LC, round_speed_after_LC]).all(1))
        assert (len(index) <= 1)
        index = index[0]

        if len(index):
            CF_challenge_array = global_val.CF_challenge_value[index.item(), :]

            if action_idx == 0:
                front_veh_after_LC = f0
            elif action_idx == 1:
                front_veh_after_LC = f2

            if front_veh_after_LC:
                predict_v, predict_x, predict_lane_id = front_veh_after_LC.velocity, front_veh_after_LC.position[0] + front_veh_after_LC.velocity * simulation_resolution, \
                                                        front_veh_after_LC.lane_index[2]
                predict_front_veh_after_LC = BV_model(env.road, env.road.network.get_lane(("a", "b", predict_lane_id)).position(predict_x, 0), 0, predict_v)
                predict_self_v, predict_self_x, predict_self_lane_id = bv.velocity, bv.position[0] + bv.velocity * simulation_resolution, front_veh_after_LC.lane_index[2]
                predict_self_after_LC = BV_model(env.road, env.road.network.get_lane(("a", "b", predict_self_lane_id)).position(predict_self_x, 0), 0, predict_self_v)
            else:
                predict_front_veh_after_LC = None
                predict_self_after_LC = bv

            obs_bv_after_LC = [predict_front_veh_after_LC, None, None, None, None, None]
            _, long_possi_after_LC_array = predict_self_after_LC.Longitudinal_NDD(obs_bv_after_LC, modify_flag=False)
            challenge = np.sum(long_possi_after_LC_array * CF_challenge_array)

    try:
        assert (challenge <= 1.00001)
    except:
        print(challenge)
        print(long_possi_after_LC_array)
        print(CF_challenge_array)
        raise ValueError("Challenge")
    return challenge


def _BV_tail_following_challenge(bv, v, r, rr, env):
    """
    Calculate the challenge when BV LC to AV's lane and then follow the AV.

    Args:
        bv: the BV.
        v: velocity of the BV.
        r: the range (spacing) between BV and the following AV.
        rr: the range rate (relative speed) between BV and the following AV.
        env: the simulation environment.

    Returns:
        float: the challenge value of this situation.

    """
    challenge = 0

    cav = env.vehicle
    # Assume the CAV is decelerate at maximum at this moment.
    cav_v, cav_acc, cav_x = cav.velocity, global_val.acc_low, cav.position[0]
    predict_cav_v_after_LC = np.clip(cav_v + cav_acc * simulation_resolution, global_val.v_low, global_val.v_high)
    v_after_LC, rr_after_LC = v, predict_cav_v_after_LC - v
    r_after_LC = r + rr * simulation_resolution

    if r_after_LC <= 0:  # Crash during the LC.
        challenge = 1
    else:  # The challenge in car-following situation after LC.
        round_speed_after_LC, _, round_r_after_LC, _, round_rr_after_LC, _ = _round_data(bv, v_after_LC, r_after_LC, rr_after_LC)

        index = np.where((global_val.BV_CF_state_value == [round_r_after_LC, round_rr_after_LC, round_speed_after_LC]).all(1))
        assert (len(index) <= 1)
        index = index[0]

        if len(index):
            BV_CF_challenge_array = global_val.BV_CF_challenge_value[index.item(), :]

            predict_cav_v, predict_cav_lane_id = predict_cav_v_after_LC, cav.lane_index[2]
            predict_cav_x = cav_x + utils.cal_dis_with_start_end_speed(cav_v, predict_cav_v_after_LC, cav_acc, time_interval=global_val.simulation_resolution)
            predict_front_veh_after_LC = BV_model(env.road, env.road.network.get_lane(("a", "b", predict_cav_lane_id)).position(predict_cav_x, 0), 0, predict_cav_v)

            predict_self_v, predict_self_x, predict_self_lane_id = bv.velocity, bv.position[0] + bv.velocity * simulation_resolution, cav.lane_index[2]
            predict_self_after_LC = BV_model(env.road, env.road.network.get_lane(("a", "b", predict_self_lane_id)).position(predict_self_x, 0), 0, predict_self_v)

            obs_bv_after_LC = [predict_front_veh_after_LC, None, None, None, None, None]
            _, long_possi_after_LC_array = predict_self_after_LC.Longitudinal_NDD(obs_bv_after_LC, modify_flag=False)
            challenge = np.sum(long_possi_after_LC_array * BV_CF_challenge_array)

    try:
        assert (challenge <= 1.00001)
    except:
        print(challenge)
        print(long_possi_after_LC_array)
        print(BV_CF_challenge_array)
        raise ValueError("BV CF Challenge")
    return challenge


def _BV_accelerate_challenge(bv, v, r, rr):
    """
    Calculate the challenge when the AV lane-change to BV's lane (BV is on the adjacent lane and behind the AV).

    Args:
        bv: the BV.
        v: velocity of the BV.
        r: the range (spacing) between BV and the following AV.
        rr: the range rate (relative speed) between BV and the following AV.

    Returns:
        float: the challenge value of this situation.

    """
    BV_CF_challenge_array = np.zeros((len(global_val.BV_ACTIONS) - 2), dtype=float)
    round_speed, _, round_r, _, round_rr, _ = _round_data(bv, v, r, rr)

    index = np.where((global_val.BV_CF_state_value == [round_r, round_rr, round_speed]).all(1))
    assert (len(index) <= 1)
    index = index[0]

    if len(index):
        BV_CF_challenge_array = global_val.BV_CF_challenge_value[index.item(), :]
    return BV_CF_challenge_array


def _lane_conflict_challenge(bv, obs_bv, action_idx, env):
    """
    Calculate the challenge value when the BV and SV doing a LC simultaneously to the same middle lane. It is composed by two parts
    - challenge during the LC (crash happens during the LC process).
    - challenge after the LC (crash happens in the car-following situation after the LC process).

    Args:
        bv: the BV.
        obs_bv: BV observation.
        action_idx: 0 or 1. 0 stands for left turn, 1 stands for right turn.
        env: the simulation environment.

    Returns:
        float: the challenge value of this situation.
    """
    challenge = 0

    _, _, f0, _, f2, _ = obs_bv
    if bv.position[0] >= env.vehicle.position[0]:  # BV is in front of the AV, AV car-following the BV after LC.
        v = bv.velocity
        r = bv.position[0] - env.vehicle.position[0] - bv.LENGTH
        rr = bv.velocity - env.vehicle.velocity

        v_after_LC, rr_after_LC = v, rr
        r_after_LC = r + rr * simulation_resolution

        if r <= 0 or r_after_LC <= 0:
            challenge = 1
            return challenge
        else:
            round_speed_after_LC, _, round_r_after_LC, _, round_rr_after_LC, _ = _round_data(bv, v_after_LC, r_after_LC, rr_after_LC)

            index = np.where((global_val.CF_state_value == [round_r_after_LC, round_rr_after_LC, round_speed_after_LC]).all(1))
            assert (len(index) <= 1)
            index = index[0]

            if len(index):
                CF_challenge_array = global_val.CF_challenge_value[index.item(), :]

                if action_idx == 0:
                    front_veh_after_LC = f0
                elif action_idx == 1:
                    front_veh_after_LC = f2

                if front_veh_after_LC:
                    predict_v, predict_x, predict_lane_id = front_veh_after_LC.velocity, front_veh_after_LC.position[0] + front_veh_after_LC.velocity * simulation_resolution, \
                                                            front_veh_after_LC.lane_index[2]
                    predict_front_veh_after_LC = BV_model(env.road, env.road.network.get_lane(("a", "b", predict_lane_id)).position(predict_x, 0), 0, predict_v)
                    predict_self_v, predict_self_x, predict_self_lane_id = bv.velocity, bv.position[0] + bv.velocity * simulation_resolution, front_veh_after_LC.lane_index[2]
                    predict_self_after_LC = BV_model(env.road, env.road.network.get_lane(("a", "b", predict_self_lane_id)).position(predict_self_x, 0), 0, predict_self_v)
                else:
                    predict_front_veh_after_LC = None
                    predict_self_after_LC = bv

                obs_bv_after_LC = [predict_front_veh_after_LC, None, None, None, None, None]
                _, long_possi_after_LC_array = predict_self_after_LC.Longitudinal_NDD(obs_bv_after_LC, modify_flag=False)
                challenge = np.sum(long_possi_after_LC_array * CF_challenge_array)
    else:  # BV is in behind the AV, BV car-following the AV after LC.
        r = env.vehicle.position[0] - bv.position[0] - bv.LENGTH
        rr = env.vehicle.velocity - bv.velocity

        r_after_LC = r + rr * simulation_resolution

        if r <= 0 or r_after_LC <= 0:
            challenge = 1
            return challenge
        else:
            cav = env.vehicle
            v_after_LC, rr_after_LC = bv.velocity, rr
            r_after_LC = r + rr * simulation_resolution

            round_speed_after_LC, _, round_r_after_LC, _, round_rr_after_LC, _ = _round_data(bv, v_after_LC, r_after_LC, rr_after_LC)

            index = np.where((global_val.BV_CF_state_value == [round_r_after_LC, round_rr_after_LC, round_speed_after_LC]).all(1))
            assert (len(index) <= 1)
            index = index[0]

            if len(index):
                BV_CF_challenge_array = global_val.BV_CF_challenge_value[index.item(), :]

                if action_idx == 0:
                    predict_cav_lane_id = bv.lane_index[2] - 1
                elif action_idx == 1:
                    predict_cav_lane_id = bv.lane_index[2] + 1

                predict_cav_v = cav.velocity
                predict_cav_x = cav.position[0] + cav.velocity * simulation_resolution,
                predict_front_veh_after_LC = BV_model(env.road, env.road.network.get_lane(("a", "b", predict_cav_lane_id)).position(predict_cav_x, 0), 0, predict_cav_v)

                predict_self_v, predict_self_x, predict_self_lane_id = bv.velocity, bv.position[0] + bv.velocity * simulation_resolution, predict_cav_lane_id
                predict_self_after_LC = BV_model(env.road, env.road.network.get_lane(("a", "b", predict_self_lane_id)).position(predict_self_x, 0), 0, predict_self_v)

                obs_bv_after_LC = [predict_front_veh_after_LC, None, None, None, None, None]
                _, long_possi_after_LC_array = predict_self_after_LC.Longitudinal_NDD(obs_bv_after_LC, modify_flag=False)
                challenge = np.sum(long_possi_after_LC_array * BV_CF_challenge_array)

    try:
        assert (challenge <= 1.00001)
    except Exception:
        print(challenge)
        print(long_possi_after_LC_array)
        print(CF_challenge_array)
        raise ValueError("Challenge")
    return challenge


def _get_Surrogate_CAV_action_probability(env):
    """
    Obtain the AV action probability using surrogate model.

    Args:
        env: the simulation environment.

    Returns:
        (float, float, float):
        - CAV_left_prob - CAV left turn probability.
        - CAV_still_prob - CAV going straight probability.
        - CAV_right_prob - CAV right turn probability.

    """
    CAV_left_prob, CAV_still_prob, CAV_right_prob = 0, global_val.SM_epsilon_still_probability, 0
    left_gain, right_gain = 0, 0
    left_LC_safety_flag, right_LC_safety_flag = False, False
    # CAV will do lane change or not?
    CAV_current_lane_id = env.vehicle.lane_index[2]
    for lane_index in env.vehicle.road.network.side_lanes(env.vehicle.lane_index):
        LC_safety_flag, gain = _Mobil_surraget_model(env.vehicle, lane_index)
        if gain is not None:
            if lane_index[2] > CAV_current_lane_id:
                right_gain = np.clip(gain, 0., None)
                right_LC_safety_flag = LC_safety_flag
            elif lane_index[2] < CAV_current_lane_id:
                left_gain = np.clip(gain, 0., None)
                left_LC_safety_flag = LC_safety_flag
    assert (left_gain >= 0 and right_gain >= 0)

    # epsilon LC probability if no safety issue and feasible for LC
    CAV_left_prob += global_val.SM_epsilon_lane_change_probability * left_LC_safety_flag
    CAV_right_prob += global_val.SM_epsilon_lane_change_probability * right_LC_safety_flag

    max_remaining_LC_probability = 1 - global_val.SM_epsilon_still_probability - CAV_left_prob - CAV_right_prob
    total_gain = left_gain + right_gain
    obtained_LC_probability_for_sharing = np.clip(utils.remap(total_gain, [0, global_val.SM_MOBIL_max_gain_threshold], [0, max_remaining_LC_probability]), 0, max_remaining_LC_probability)
    CAV_still_prob += (max_remaining_LC_probability - obtained_LC_probability_for_sharing)

    if total_gain > 0:
        CAV_left_prob += obtained_LC_probability_for_sharing * (left_gain / (left_gain + right_gain))
        CAV_right_prob += obtained_LC_probability_for_sharing * (right_gain / (left_gain + right_gain))

    assert (0.99999 <= (CAV_left_prob + CAV_still_prob + CAV_right_prob) <= 1.0001)

    return CAV_left_prob, CAV_still_prob, CAV_right_prob


def Decompose_decision(bv, obs_bv, AV_SM_prob, env):
    """
    Calculate criticality, action, weight (importance sampling likelihood), sampled action naturalistic probability, sampled action importance function probability for a specific BV.
    The criticality is calculated based on exposure frequency and maneuver challenge.

    Args:
        bv: the specific BV to evaluate.
        obs_bv: the BV's observation of surrounding vehicles.
        AV_SM_prob: AV action probability from surrogate model.
        env: the simulation environment.

    Returns:
        - bv_criticality - the criticality of the BV consider its all potential actions.
        - bv_action_idx - sampled BV action index according to importance function if its criticality greater than the pre-defined threshold. Otherwise, it is None.
        - weight - the weight (importance sampling likelihood) of the sampled action if bv_action_idx is applicable. Otherwise, None.
        - ndd_possi - the sampled action naturalistic probability if bv_action_idx is applicable. Otherwise, None.
        - critical_possi - the sampled action importance function probability if bv_action_idx is applicable. Otherwise, None.

    """

    # CAV will do lane change or not?
    CAV_current_lane_id = env.vehicle.lane_index[2]
    CAV_left_prob, CAV_still_prob, CAV_right_prob = AV_SM_prob

    # f1 denotes vehicle in front of the BV on the same lane, f0 denotes vehicle in front on the left lane, f2 denotes vehicle in front on the right lane. r is for vehicles behind the BV.
    f1, r1, f0, r0, f2, r2 = obs_bv
    bv_criticality, bv_action_idx, weight, ndd_possi, critical_possi = -np.inf, None, None, None, None

    possi_array = get_NDD_possi(bv, obs_bv)  # Get BV naturalistic behavior probability at the current moment.
    if (0.99999 <= possi_array[0] <= 1) or (0.99999 <= possi_array[1] <= 1):  # If lane change prob = 1, then not control.
        return bv_criticality, bv_action_idx, weight, ndd_possi, critical_possi

    if r1 == env.vehicle:  # When the AV is behind the BV (BV and AV in the same lane).
        criticality_array = np.zeros((len(global_val.BV_ACTIONS)), dtype=float)  # The criticality for each action.

        v = bv.velocity  # BV speed.
        r = bv.position[0] - env.vehicle.position[0] - bv.LENGTH  # range (spacing).
        rr = bv.velocity - env.vehicle.velocity  # range rate (relative speed).

        # Left turn criticality:
        if CAV_left_prob != 0 and possi_array[0] != 0:
            challenge = _lane_change_with_CAV_challenge(bv, v, r, rr, obs_bv, action_idx=0, env=env)
            criticality_array[0] = challenge * possi_array[0] * CAV_left_prob  # criticality of the left lane change maneuver.

        # Right turn criticality:
        if CAV_right_prob != 0 and possi_array[1] != 0:
            challenge = _lane_change_with_CAV_challenge(bv, v, r, rr, obs_bv, action_idx=1, env=env)
            criticality_array[1] = challenge * possi_array[1] * CAV_right_prob

        # Go stright criticality:
        if np.sum(possi_array[2:]) != 0:
            CF_challenge_array = _hard_brake_challenge(bv, v, r, rr)  # challenge of each longitudinal acceleration.
            criticality_array[2:] = CF_challenge_array * (possi_array[2:]) * CAV_still_prob  # criticality of each longitudinal acceleration.

        bv_criticality = np.sum(criticality_array)  # The criticality of the BV.
        if bv_criticality > criticality_threshold:
            bv_action_idx, weight, ndd_possi, critical_possi = _sample_critical_action(criticality_array, bv_criticality, possi_array)

    elif (r0 == env.vehicle) or (r2 == env.vehicle):  # When the AV is behind the BV (AV is on the adjacent lane of the BV).
        criticality_array = np.zeros((len(global_val.BV_ACTIONS)), dtype=float)

        v = bv.velocity
        r = bv.position[0] - env.vehicle.position[0] - bv.LENGTH
        rr = bv.velocity - env.vehicle.velocity

        # Left turn criticality:
        if r0 == env.vehicle and possi_array[0] != 0:
            challenge = _cut_in_challenge(bv, v, r, rr, obs_bv, action_idx=0, env=env)
            criticality_array[0] = challenge * possi_array[0] * CAV_still_prob

        # Right turn criticality:
        if r2 == env.vehicle and possi_array[1] != 0:
            challenge = _cut_in_challenge(bv, v, r, rr, obs_bv, action_idx=1, env=env)
            criticality_array[1] = challenge * possi_array[1] * CAV_still_prob

        # Go stright criticality: AV LC to the BV's lane.
        if r0 == env.vehicle and CAV_right_prob != 0:
            CF_challenge_array = _hard_brake_challenge(bv, v, r, rr)
            criticality_array[2:] = CF_challenge_array * (possi_array[2:]) * CAV_right_prob
        if r2 == env.vehicle and CAV_left_prob != 0:
            CF_challenge_array = _hard_brake_challenge(bv, v, r, rr)
            criticality_array[2:] = CF_challenge_array * (possi_array[2:]) * CAV_left_prob

        bv_criticality = np.sum(criticality_array)
        if bv_criticality > criticality_threshold:
            bv_action_idx, weight, ndd_possi, critical_possi = _sample_critical_action(criticality_array, bv_criticality, possi_array)

    elif (CAV_current_lane_id == 0 and bv.lane_index[2] == 2) or (CAV_current_lane_id == 2 and bv.lane_index[2] == 0):  # There is a lane between AV and BV.
        criticality_array = np.zeros((len(global_val.BV_ACTIONS)), dtype=float)

        # Left turn criticality: both the AV and BV LC to the same middle lane.
        if (CAV_current_lane_id == 0 and bv.lane_index[2] == 2) and CAV_right_prob != 0 and possi_array[0] != 0:
            challenge = _lane_conflict_challenge(bv, obs_bv, action_idx=0, env=env)
            criticality_array[0] = challenge * possi_array[0] * CAV_right_prob

        # Right turn criticality: both the AV and BV LC to the same middle lane.
        elif (CAV_current_lane_id == 2 and bv.lane_index[2] == 0) and CAV_left_prob != 0 and possi_array[1] != 0:
            challenge = _lane_conflict_challenge(bv, obs_bv, action_idx=1, env=env)
            criticality_array[1] = challenge * possi_array[1] * CAV_left_prob

        bv_criticality = np.sum(criticality_array)
        if bv_criticality > criticality_threshold:
            bv_action_idx, weight, ndd_possi, critical_possi = _sample_critical_action(criticality_array, bv_criticality, possi_array)

    elif (f0 == env.vehicle) or (f2 == env.vehicle):  # When the BV is behind the AV (AV is on the adjacent lane of the BV).
        criticality_array = np.zeros((len(global_val.BV_ACTIONS)), dtype=float)

        v = bv.velocity
        r = env.vehicle.position[0] - bv.position[0] - bv.LENGTH
        rr = env.vehicle.velocity - bv.velocity

        # Left turn criticality:
        if f0 == env.vehicle and possi_array[0] != 0:
            challenge = _BV_tail_following_challenge(bv, v, r, rr, env)
            criticality_array[0] = challenge * possi_array[0] * CAV_still_prob

        # Right turn criticality:
        if f2 == env.vehicle and possi_array[1] != 0:
            challenge = _BV_tail_following_challenge(bv, v, r, rr, env)
            criticality_array[1] = challenge * possi_array[1] * CAV_still_prob

        # Go stright criticality:
        if f0 == env.vehicle and CAV_right_prob != 0:
            BV_CF_challenge_array = _BV_accelerate_challenge(bv, v, r, rr)
            criticality_array[2:] = BV_CF_challenge_array * (possi_array[2:]) * CAV_right_prob
        if f2 == env.vehicle and CAV_left_prob != 0:
            BV_CF_challenge_array = _BV_accelerate_challenge(bv, v, r, rr)
            criticality_array[2:] = BV_CF_challenge_array * (possi_array[2:]) * CAV_left_prob

        bv_criticality = np.sum(criticality_array)
        if bv_criticality > criticality_threshold:
            bv_action_idx, weight, ndd_possi, critical_possi = _sample_critical_action(criticality_array, bv_criticality, possi_array)

    elif f1 == env.vehicle:  # When the BV is behind the AV (BV and AV in the same lane).
        criticality_array = np.zeros((len(global_val.BV_ACTIONS)), dtype=float)

        v = bv.velocity
        r = env.vehicle.position[0] - bv.position[0] - bv.LENGTH
        rr = env.vehicle.velocity - bv.velocity

        # Left turn criticality:
        if CAV_left_prob != 0 and possi_array[0] != 0:
            challenge = _BV_behind_lane_change_with_CAV_challenge(bv, v, r, rr, action_idx=0, env=env)
            criticality_array[0] = challenge * possi_array[0] * CAV_left_prob

        # Right turn criticality:
        if CAV_right_prob != 0 and possi_array[1] != 0:
            challenge = _BV_behind_lane_change_with_CAV_challenge(bv, v, r, rr, action_idx=1, env=env)
            criticality_array[1] = challenge * possi_array[1] * CAV_right_prob

        # Go stright criticality:
        if np.sum(possi_array[2:]) != 0:
            BV_CF_challenge_array = _BV_accelerate_challenge(bv, v, r, rr)
            criticality_array[2:] = BV_CF_challenge_array * (possi_array[2:]) * CAV_still_prob

        bv_criticality = np.sum(criticality_array)
        if bv_criticality > criticality_threshold:
            bv_action_idx, weight, ndd_possi, critical_possi = _sample_critical_action(criticality_array, bv_criticality, possi_array)

    if weight:
        weight, ndd_possi, critical_possi = weight.item(), ndd_possi.item(), critical_possi.item()
    return bv_criticality, bv_action_idx, weight, ndd_possi, critical_possi


def _Mobil_surraget_model(cav, lane_index):
    """
    Calculate CAV lane change gain using MOBIL surrogate model.

    Args:
        cav: the AV.
        lane_index: the target lane index.

    Returns:
        (bool, float):
        - safety flag: whether AV will crash immediately after doing LC.
        - gain: the gain of performing LC.

    """
    gain = None

    # Is the maneuver unsafe for the new following vehicle?
    new_preceding, new_following = cav.road.neighbour_vehicles(cav, lane_index)

    # Check whether will crash immediately
    r_new_preceding, r_new_following = 99999, 99999
    if new_preceding:
        r_new_preceding = new_preceding.position[0] - cav.position[0] - cav.LENGTH
    if new_following:
        r_new_following = cav.position[0] - new_following.position[0] - cav.LENGTH
    if r_new_preceding <= 0 or r_new_following <= 0:
        return False, gain

    new_following_a = cav.acceleration(ego_vehicle=new_following, front_vehicle=new_preceding)
    new_following_pred_a = cav.acceleration(ego_vehicle=new_following, front_vehicle=cav)

    old_preceding, old_following = cav.road.neighbour_vehicles(cav)
    self_pred_a = cav.acceleration(ego_vehicle=cav, front_vehicle=new_preceding)

    # Is there an acceleration advantage for me and/or my followers to change lane?
    self_a = cav.acceleration(ego_vehicle=cav, front_vehicle=old_preceding)
    old_following_a = cav.acceleration(ego_vehicle=old_following, front_vehicle=cav)
    old_following_pred_a = cav.acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
    gain = self_pred_a - self_a + global_val.SM_POLITENESS * (new_following_pred_a - new_following_a + old_following_pred_a - old_following_a)
    return True, gain
