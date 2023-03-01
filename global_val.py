from bidict import bidict
import os
import numpy as np
import scipy.io

# =========== Simulation parameters ==========
simulation_resolution = 1  # The time interval(resolution) of the simulation platform (second)
# =========== Vehicle parameters =============
LENGTH, WIDTH = 5.0, 2.0
# =========== Highway parameters =============
HIGHWAY_LENGTH = 1200
EXIT_LENGTH = 800  # AV will run 400m each test (from 400 to 800).
v_min, v_max = 20, 40  # m/s vehicles speed limit
a_min, a_max = -4, 2  # m/s^2

# =========== Initialization of env ===========
initial_CAV_speed = 30  # m/s AV initial speed
initial_CAV_position = 400  # AV initial position

# Each element in presum_list_forward is the joint distribution of range and range rate at different subject vehicle velocity used for environment initialization
presum_list_forward = np.load("./Data/NDD_DATA/Initialization/Optimized_presum_list_forward.npy")
CF_percent = 0.6823141  # The probability a vehicle is car-following, 1-CF_percent is the probability of free-flow.
ff_dis = 120  # dis for ff
gen_length = 1200  # stop generate BV on the highway after this length
random_veh_pos_buffer_start, random_veh_pos_buffer_end = 0, 50
speed_CDF = list(np.load("./Data/NDD_DATA/Initialization/speed_CDF.npy"))  # the empirical speed distribution. Used for initialization.

# =========== Naturalistic Driving Model Data ===========
bv_obs_range = 120  # BV obs range.
cav_obs_range = 120  # AV obs range.
round_rule = "Round_to_closest"
# Naturalistic longitudinal behavior models: Car-following, Free-flow.
CF_pdf_array = np.load("./Data/NDD_DATA/CF/CF_pdf_array.npy")
FF_pdf_array = np.load("./Data/NDD_DATA/FF/FF_pdf_array.npy")
# Naturalistic lane-changing behavior models (for situations).
SLC_pdf = np.load("./Data/NDD_DATA/LC/Lane_change_one_adjacent_vehicle_pdf.npy")  # Lane change (One adjacent vehicle)
DLC_pdf = np.load("./Data/NDD_DATA/LC/Lane_change_two_adjacent_vehicles_pdf.npy")  # Lane change (Two adjacent vehicles)
OL_pdf = np.load("./Data/NDD_DATA/LC/Lane_change_no_adjacent_vehicle_pdf.npy")  # Lane change (No adjacent vehicle)
CI_pdf = np.load("./Data/NDD_DATA/LC/Lane_change_cut-in_pdf.npy")  # Lane change (Cut in)
print("================Load Action NDD data finished!=================")

# Discretized state information corresponding to the naturalistic driving behavior models.
r_low, r_high, rr_low, rr_high, v_low, v_high, acc_low, acc_high = 0, 115, -10, 8, 20, 40, -4, 2  # m, m, m/s, m/s, m/s, m/s, m/s^2, m/s^2.
r_step, rr_step, v_step, acc_step = 1, 1, 1, 0.2  # discretized resolution. m, m/s, m/s, m/s^2.
r_to_idx_dic, rr_to_idx_dic, v_to_idx_dic, v_back_to_idx_dic, acc_to_idx_dic = bidict(), bidict(), bidict(), bidict(), bidict()  # Two way dictionary, map between state and its
# index.
speed_list, r_list, rr_list = list(range(v_low, v_high + v_step, v_step)), list(range(r_low, r_high + r_step, r_step)), list(
    range(rr_low, rr_high + rr_step, rr_step))
num_r, num_rr, num_v, num_acc = CF_pdf_array.shape
for i in range(num_r): r_to_idx_dic[list(range(r_low, r_high + r_step, r_step))[i]] = i
for j in range(num_rr): rr_to_idx_dic[list(range(rr_low, rr_high + rr_step, rr_step))[j]] = j
for k in range(num_v): v_to_idx_dic[list(range(v_low, v_high + v_step, v_step))[k]] = k
for m in range(num_acc): acc_to_idx_dic[list(np.arange(acc_low, acc_high + acc_step, acc_step))[m]] = m

# Lane-change state
lc_v_low, lc_v_high, lc_v_num = 20, 40, 21
lc_rf_low, lc_rf_high, lc_rf_num = 0, 115, 116
lc_rrf_low, lc_rrf_high, lc_rrf_num = -10, 8, 19
lc_re_low, lc_re_high, lc_re_num = 0, 115, 116
lc_rre_low, lc_rre_high, lc_rre_num = -10, 8, 19
lc_v_list, lc_r1_list, lc_r2_list, lc_rr1_list, lc_rr2_list = list(np.linspace(lc_v_low, lc_v_high, num=lc_v_num)), list(np.linspace(lc_rf_low, lc_rf_high, num=lc_rf_num)), \
                                                              list(np.linspace(lc_re_low, lc_re_high, num=lc_re_num)), list(np.linspace(lc_rrf_low, lc_rrf_high, num=lc_rrf_num)), \
                                                              list(np.linspace(lc_rre_low, lc_rre_high, num=lc_rre_num))

# =========== BV and CAV action space ================
BV_ACTIONS = {0: 'LANE_LEFT',
              1: 'LANE_RIGHT'}
num_acc = int(((acc_high - acc_low) / acc_step) + 1)
num_non_acc = len(BV_ACTIONS)
for i in range(num_acc):
    acc = acc_to_idx_dic.inverse[i]
    BV_ACTIONS[i + num_non_acc] = str(acc)

# AV
ACTIONS = {0: 'LANE_LEFT',
           1: 'LANE_RIGHT'}
num_acc = int(((acc_high - acc_low) / acc_step) + 1)
num_non_acc = len(ACTIONS)
for i in range(num_acc):
    acc = acc_to_idx_dic.inverse[i]
    ACTIONS[i + num_non_acc] = str(acc)

CAV_acc_low, CAV_acc_high, CAV_acc_step = -4, 2, 0.2
num_CAV_acc = int((CAV_acc_high - CAV_acc_low) / CAV_acc_step + 1)
CAV_acc_to_idx_dic = bidict()
for i in range(num_CAV_acc):
    CAV_acc_to_idx_dic[list(np.arange(CAV_acc_low, CAV_acc_high + CAV_acc_step, CAV_acc_step))[i]] = i

# =========== NDD ENV para ============
safety_guard_enabled_flag_IDM = True
longi_safety_buffer, lateral_safety_buffer = 2, 2  # The safety buffer used to longitudinal and lateral safety guard

Stochastic_IDM_threshold = 1e-10

enable_One_lead_LC = True
enable_Single_LC = True
enable_Double_LC = True
enable_Cut_in_LC = True

ignore_adj_veh_prob, min_r_ignore = 1e-2, 5
ignore_lane_conflict_prob = 1e-4

# ============= NADE calculate challenge data ================
CF_state_value = scipy.io.loadmat("./Data/CALCULATE_CHALLENGE/CF/" + "dangerous_state_table.mat")["dangerous_state_table"]
CF_challenge_value = scipy.io.loadmat("./Data/CALCULATE_CHALLENGE/CF/" + "Q_table.mat")["Q_table_little"]

BV_CF_state_value = np.load("./Data/CALCULATE_CHALLENGE/BV_CF/BV_CF_state_value.npy")
BV_CF_challenge_value = np.load("./Data/CALCULATE_CHALLENGE/BV_CF/BV_CF_challenge_value.npy")

episode = 0  # The episode number of the simulation

# ============= CAV surrogate model parameters =============
# Longitudinal behavior: stochastic IDM
SM_IDM_COMFORT_ACC_MAX = 2.0  # [m/s2]  2
SM_IDM_COMFORT_ACC_MIN = -4.0  # [m/s2]  -4
SM_IDM_DISTANCE_WANTED = 5.0  # [m]  5
SM_IDM_TIME_WANTED = 1.5  # [s]  1.5
SM_IDM_DESIRED_VELOCITY = 35  # [m/s]
SM_IDM_DELTA = 4.0  # []
# Lateral behavior: stochastic MOBIL
SM_epsilon_still_probability = 0.1  # 10% keep still even if Mobil model sets to do lane change
SM_POLITENESS = 0.
SM_LANE_CHANGE_MIN_ACC_GAIN = 0.2
SM_LANE_CHANGE_MAX_BRAKING_IMPOSED = 2
SM_epsilon_lane_change_probability = 1e-8
SM_MOBIL_max_gain_threshold = 1  # m/s^2 when gain is greater than this, the LC probability will be maximum
