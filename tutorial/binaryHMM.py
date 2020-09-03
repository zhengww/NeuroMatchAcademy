# %%

import numpy as np
import matplotlib.pyplot as plt

# %%

def policy(threshold, bel, loc):
    if loc == 0:
        if bel[1]  >= threshold:
            act = 1
        else:
            act = 0
    else:  # loc = 1
        if bel[0] >= threshold:
            act = 1
        else:
            act = 0

    return act

def generateProcess(params):

    T, p_sw, q_high, q_low, cost_sw, threshold = params
    world_state = np.zeros((2, T), int)  # value :1: good box; 0: bad box
    loc = np.zeros(T, int)  # 0: left box               1: right box
    obs = np.zeros(T, int)  # 0: did not get food        1: get food
    act = np.zeros(T, int)  # 0 : stay                   1: switch and get food from the other side
    bel = np.zeros((2, T), float)  # the probability that the left box has food,
    # then the probability that the second box has food is 1-b


    p = np.array([1 - p_sw, p_sw])  # transition probability to good state
    q = np.array([q_low, q_high])
    q_mat = np.array([[1 - q_high, q_high], [1 - q_low, q_low]])

    for t in range(T):
        if t == 0:
            world_state[0, t] = 1    # good box
            world_state[1, t] = 1 - world_state[0, t]
            loc[t] = 0
            obs[t] = 0
            bel_0 = np.random.random(1)[0]
            bel[:, t] = np.array([bel_0, 1-bel_0])

            act[t] = policy(threshold, bel[:, t], loc[t])

        else:
            world_state[0, t] = np.random.binomial(1, p[world_state[0, t - 1]])
            world_state[1, t] = 1 - world_state[0, t]

            if act[t - 1] == 0:
                loc[t] = loc[t - 1]
            else:  # after weitching, open the new box, deplete if any; then wait a usualy time
                loc[t] = 1 - loc[t - 1]

            # new observation
            obs[t] = np.random.binomial(1, q[world_state[loc[t], t-1]])

            # update belief posterior, p(s[t] | obs(0-t), act(0-t-1))
            bel_0 = (bel[0, t-1] * p_sw  + bel[1, t-1] * (1 - p_sw)) * q_mat[loc[t], obs[t]]
            bel_1 = (bel[1, t - 1] * p_sw + bel[0, t - 1] * (1 - p_sw)) * q_mat[1-loc[t], obs[t]]

            bel[0, t] = bel_0 / (bel_0 + bel_1)
            bel[1, t] = bel_1 / (bel_0 + bel_1)

            act[t] = policy(threshold, bel[:, t], loc[t])

    return bel, obs, act, world_state, loc

def value_function(obs, act, cost_sw, discount):
    T = len(obs)
    discount_time = np.array([discount ** t for t in range(T)])

    #value = (np.sum(obs) - np.sum(act) * cost_sw) / T
    value = (np.sum(np.multiply(obs, discount_time)) - np.sum(np.multiply(act, discount_time)) * cost_sw) / T

    return value

def switch_int(obs, act):
    sw_t = np.where(act == 1)[0]
    sw_int = sw_t[1:] - sw_t[:-1]

    return sw_int

def plot_dynamics(bel, obs, act, world_state, loc):
    T = len(obs)

    showlen = min(T, 100)
    startT = 0

    endT = startT + showlen
    showT = range(startT, endT)
    time_range = np.linspace(0, showlen - 1)

    fig_posterior, [ax0, ax1, ax_loc, ax2, ax3] = plt.subplots(5, 1, figsize=(15, 10))

    ax0.plot(world_state[0, showT], color='dodgerblue', markersize=10, linewidth=3.0)
    ax0.set_ylabel('Left box', rotation=360, fontsize=22)
    ax0.yaxis.set_label_coords(-0.1, 0.25)
    ax0.set_xticks(np.arange(0, showlen, 10))
    ax0.tick_params(axis='both', which='major', labelsize=18)
    ax0.set_xlim([0, showlen])


    ax3.plot(world_state[1, showT], color='dodgerblue', markersize=10, linewidth=3.0)
    ax3.set_ylabel('Right box', rotation=360, fontsize=22)
    ax3.yaxis.set_label_coords(-0.1, 0.25)
    ax3.tick_params(axis='both', which='major', labelsize=18)
    ax3.set_xlim([0, showlen])
    ax3.set_xticks(np.arange(0, showlen, 10))

    ax1.plot(bel[0, showT], color='dodgerblue', markersize=10, linewidth=3.0)
    ax1.plot(time_range, threshold * np.ones(time_range.shape), 'r--')
    ax1.yaxis.set_label_coords(-0.1, 0.25)
    ax1.set_ylabel('Belief on \n left box', rotation=360, fontsize=22)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.set_xlim([0, showlen])
    ax1.set_ylim([0, 1])
    ax1.set_xticks(np.arange(0, showlen, 10))


    ax_loc.plot(1 - loc[showT], 'g.-', markersize=12, linewidth=5, label = 'location')
    ax_loc.plot((act[showT] - .1) * .8, 'v', markersize=10, label = 'action')
    ax_loc.plot(obs[showT] * .5, '*', markersize=5, label = 'reward')
    ax_loc.legend(loc="upper right")
    ax_loc.set_xlim([0, showlen])
    ax_loc.set_ylim([0, 1])
    #ax_loc.set_yticks([])
    ax_loc.set_xticks([0, showlen])
    ax_loc.tick_params(axis='both', which='major', labelsize=18)
    labels = [item.get_text() for item in ax_loc.get_yticklabels()]
    labels[0] = 'Right'
    labels[-1] = 'Left'
    ax_loc.set_yticklabels(labels)

    ax2.plot(bel[1, showT], color='dodgerblue', markersize=10, linewidth=3.0)
    ax2.plot(time_range, threshold * np.ones(time_range.shape), 'r--')
    ax2.set_xlabel('time', fontsize=18)
    ax2.yaxis.set_label_coords(-0.1, 0.25)
    ax2.set_ylabel('Belief on  \n  right box', rotation=360, fontsize=22)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.set_xlim([0, showlen])
    ax2.set_ylim([0, 1])
    ax2.set_xticks(np.arange(0, showlen, 10))

    plt.show()

def plot_val_thre(threshold_array, value_array):
    fig_, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(threshold_array, value_array)
    ax.set_ylim([np.min(value_array), np.max(value_array)])
    ax.set_title('threshold vs value')
    ax.set_xlabel('threshold')
    ax.set_ylabel('value')
    plt.show()


T = 5000
p_sw = .95          # state transiton probability
q_high = .7
q_low = 0 #.2
cost_sw = 5 #int(1/(1-p_sw)) - 5
threshold = .8    # threshold of belief for switching
discount = 1

step = 0.2
threshold_array = np.arange(0, 1 + step, step)
value_array = np.zeros(threshold_array.shape)

for i in range(len(threshold_array)):
    threshold = threshold_array[i]
    params = [T, p_sw, q_high, q_low, cost_sw, threshold]
    bel, obs, act, world_state, loc = generateProcess(params)
    value_array[i] = value_function(obs, act, cost_sw, discount)
    sw_int = switch_int(obs, act)
    print(np.mean(sw_int))

    if threshold == 0.8:
        plot_dynamics(bel, obs, act, world_state, loc)

plot_val_thre(threshold_array, value_array)


