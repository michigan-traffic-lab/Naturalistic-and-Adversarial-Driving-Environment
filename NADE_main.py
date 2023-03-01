import sys
import argparse
import os
import yaml
from timeit import default_timer as timer

from NADE_core import select_controlled_bv_and_action  # Import NADE algorithm
import global_val
from highway_env.envs.highway_env_NDD import *  # Environment
from CAV_agent.agent import AV_RL_agent  # AV agent

start = timer()
# settings
parser = argparse.ArgumentParser()
parser.add_argument('--folder-idx', type=int, default='1', metavar='N',
                    help='Worker id of the running experiment')
parser.add_argument('--experiment-name', type=str, required=True,
                    help='The name of the experiment folder where the data will be stored')
args = parser.parse_args()


def output_whole_dict(resdict):
    """
    Print out test results. E.g., total number of episodes have accident, percentage of episodes have accident.

    Args:
        resdict: test result dictionary.

    """
    print(resdict)
    whole_num = sum(resdict.values())
    for key in resdict:
        rate = resdict[key] / whole_num
        print(key, "rate", rate)


if __name__ == '__main__':

    # Load config file
    with open('configs.yml') as file:
        try:
            configs = yaml.safe_load(file)
        except yaml.YAMLError as exception:
            print(exception)

    CAV_model = "AV2"  # AV2: using RL AV model (AV2 in the paper).
    env_config = {"min_distance": 0, "max_distance": 1200, "min_velocity": 20, "max_velocity": 40, "min_lane": 0, "max_lane": 2, "cav_observation_range": global_val.cav_obs_range,
                  "bv_observation_range": global_val.bv_obs_range, "candidate_controlled_bv_num": 8, "cav_observation_num": 10, "bv_observation_num": 10, "generate_vehicle_mode": "NDD",
                  "delete_BV_position": 1200, "CAV_model": CAV_model, "policy_frequency": 1}

    env = HighwayEnvNDD(env_config)  # Initialize simulation environment.
    if CAV_model == "AV2":
        print("Using AV2 (RL agent) as the CAV model!")
        CAV_agent = AV_RL_agent(env)
    else:
        raise NotImplementedError('{0} does not supported..., set to AV2'.format(CAV_model))

    experiment_name = args.experiment_name
    process_id = str(args.folder_idx)
    log_name = "NADE_Evaluation_log_" + experiment_name + "_" + process_id + ".txt"
    if configs['log_out_flag']:
        main_folder = os.path.join("Log", experiment_name)
    save_log_address = os.path.join(main_folder, log_name)
    if os.path.exists(main_folder) or not configs['log_out_flag']:
        pass
    else:
        os.makedirs(main_folder, exist_ok=True)

    whole_dict = {"AV-Finish-Test": 0, "AV-Crash": 0}
    for test_item in range(configs['TEST_EPISODE']):  # Main loop for each simulation episode.
        global_val.episode = test_item

        obs_and_indicator, _ = env.reset()  # Initialize and reset the simulation environment.
        obs, action_indicator = obs_and_indicator[0], obs_and_indicator[1]  # observation of the AV and the action indicator array (1 means the action is safe and 0 means
        # dangerous)
        done = False  # The flag of the end of each episode.
        if configs['render_flag']:
            env.render()

        weight_list_one_simulation, criticality_list_one_simulation, ndd_possi_list_one_simulation, critical_possi_list_one_simulation = [], [], [], []  # Log out data
        while not done:
            # NADE algorithm: Select controlled BVs and actions. Currently choose one BV that with the maximum criticality to control.
            bv_action_idx_list, weight_list, max_criticality, ndd_possi_list, critical_possi_list = select_controlled_bv_and_action(obs.bv_observation, env)
            BV_action = bv_action_idx_list

            # Log out data
            tmp_weight, tmp_ndd_possi, tmp_critical_possi = [val for val in weight_list if val is not None], [val for val in ndd_possi_list if val is not None], [val for val in
                                                                                                                                                                  critical_possi_list if val is not None]
            weight_list_one_simulation += tmp_weight if len(tmp_weight) > 0 else [1]
            ndd_possi_list_one_simulation += tmp_ndd_possi if len(tmp_ndd_possi) > 0 else [1]
            critical_possi_list_one_simulation += tmp_critical_possi if len(tmp_critical_possi) > 0 else [1]
            criticality_list_one_simulation.append(max_criticality)

            # CAV action
            action_indicator_after_lane_conflict = CAV_agent.lane_conflict_safety_check(obs.cav_observation, action_indicator.cav_indicator)
            CAV_action = CAV_agent.decision(obs.cav_observation, action_indicator_after_lane_conflict)

            action = Action(cav_action=CAV_action, bv_action=BV_action)
            obs_and_indicator, done, info, weight = env.step(action)  # Simulate one step.
            obs, action_indicator = obs_and_indicator[0], obs_and_indicator[1]
            if configs['render_flag']:
                env.render()

        whole_dict[info["scene_type"]] += 1
        if test_item % 1 == 0:
            output_whole_dict(whole_dict)

        # Log out testing result data
        if configs['log_out_flag']:
            with open(save_log_address, "a") as out_file:
                # 1: AV-Crash; 3: AV-finish
                if info["scene_type"] == "AV-Crash":
                    out_file.write(str(test_item) + "\t" + "1" + "\t" + str(weight_list_one_simulation) + "\t" + str(criticality_list_one_simulation) + "\t" + str(
                        ndd_possi_list_one_simulation) + "\t" + str(critical_possi_list_one_simulation) + "\n")
                else:
                    out_file.write(str(test_item) + "\t" + "3" + "\t" + str(weight_list_one_simulation) + "\t" + str(criticality_list_one_simulation) + "\t" + str(
                        ndd_possi_list_one_simulation) + "\t" + str(critical_possi_list_one_simulation) + "\n")
    end = timer()
    print("Time:", end - start)
