import numpy as np

class State_p_v:
    # rel_pos is the pos relative to the observing agent, which is read from agent's observation
    # abs_pos is the absolute pos in the world, which is read from env.agent.state.p_pos
    # vel is always absolute
    def __init__(self, rel_p_x, rel_p_y, abs_p_x, abs_p_y, v_x, v_y, is_dead):
        # in a 2D world
        self.rel_pos = np.zeros(2)
        self.abs_pos = np.zeros(2)
        self.vel = np.zeros(2)
        # self.pos_x = p_x
        # self.pos_y = p_y
        # self.vel_x = v_x
        # self.vel_y = v_y
        self.rel_pos[0] = rel_p_x
        self.rel_pos[1] = rel_p_y
        self.abs_pos[0] = abs_p_x
        self.abs_pos[1] = abs_p_y
        self.vel[0] = v_x
        self.vel[1] = v_y
        self.is_dead = is_dead

def get_states(agent, env, state, num_allies, num_adversaries):
    # numbers are hard coded as 3 now
    num_friends = num_allies - 1
    num_agents = num_allies + num_adversaries

    self_state = State_p_v(state[2], state[3], # rel pos
                           state[2], state[3], # abs pos
                           state[0], state[1], # vel
                           False)

    friend_states = [State_p_v(0, 0, 0, 0, 0, 0, False) for i in range(num_friends)]
    adv_states = [State_p_v(0, 0, 0, 0, 0, 0, False) for i in range(num_adversaries)]

    '''get relative pos and vel from state'''
    friend_pos_idx_start = 4
    friend_vel_idx_start = 4 + (num_agents - 1) * 2
    for friend_i in range(num_friends):
        friend_states[friend_i].rel_pos[0] = state[friend_pos_idx_start + friend_i * 2]
        friend_states[friend_i].rel_pos[1] = state[friend_pos_idx_start + friend_i * 2 + 1]
        friend_states[friend_i].vel[0] = state[friend_vel_idx_start + friend_i * 2]
        friend_states[friend_i].vel[1] = state[friend_vel_idx_start + friend_i * 2 + 1]

    adv_pos_idx_start = friend_pos_idx_start + num_friends * 2
    adv_vel_idx_start = friend_vel_idx_start + num_friends * 2
    for adv_i in range(num_adversaries):
        adv_states[adv_i].rel_pos[0] = state[adv_pos_idx_start + adv_i * 2]
        adv_states[adv_i].rel_pos[1] = state[adv_pos_idx_start + adv_i * 2 + 1]
        adv_states[adv_i].vel[0] = state[adv_vel_idx_start + adv_i * 2]
        adv_states[adv_i].vel[1] = state[adv_vel_idx_start + adv_i * 2 + 1]

    # adv_pos_idx_start = 4
    # adv_vel_idx_start = 4 + (num_agents - 1) * 2
    # for adv_i in range(num_adversaries):
    #     adv_states[adv_i].rel_pos[0] = state[adv_pos_idx_start + adv_i * 2]
    #     adv_states[adv_i].rel_pos[1] = state[adv_pos_idx_start + adv_i * 2 + 1]
    #     adv_states[adv_i].vel[0] = state[adv_vel_idx_start + adv_i * 2]
    #     adv_states[adv_i].vel[1] = state[adv_vel_idx_start + adv_i * 2 + 1]
    #
    # friend_pos_idx_start = adv_pos_idx_start + num_adversaries * 2
    # friend_vel_idx_start = adv_vel_idx_start + num_adversaries * 2
    # for friend_i in range(num_friends):
    #     friend_states[friend_i].rel_pos[0] = state[friend_pos_idx_start + friend_i * 2]
    #     friend_states[friend_i].rel_pos[1] = state[friend_pos_idx_start + friend_i * 2 + 1]
    #     friend_states[friend_i].vel[0] = state[friend_vel_idx_start + friend_i * 2]
    #     friend_states[friend_i].vel[1] = state[friend_vel_idx_start + friend_i * 2 + 1]

    '''get absolute pos and is_dead from env'''
    adv_i = 0
    friend_i = 0
    for other in env.agents:
        if other is agent:
            self_state.is_dead = other.death
            self_state.abs_pos[0] = other.state.p_pos[0]
            self_state.abs_pos[1] = other.state.p_pos[1]
            # print("self_state ", self_state.rel_pos, self_state.abs_pos, self_state.vel, self_state.is_dead)
        elif other.adversary != agent.adversary:
            adv_states[adv_i].is_dead = other.death
            adv_states[adv_i].abs_pos[0] = other.state.p_pos[0]
            adv_states[adv_i].abs_pos[1] = other.state.p_pos[1]
            adv_i += 1
        else:
            friend_states[friend_i].is_dead = other.death
            friend_states[friend_i].abs_pos[0] = other.state.p_pos[0]
            friend_states[friend_i].abs_pos[1] = other.state.p_pos[1]
            friend_i += 1

    # for adv_i in range(num_adversaries):
        # print("adv_states ", adv_i, adv_states[adv_i].rel_pos, adv_states[adv_i].abs_pos,
        #       adv_states[adv_i].vel, adv_states[adv_i].is_dead)
    # for friend_i in range(num_friends):
        # print("friend_states ", friend_i, friend_states[friend_i].rel_pos, friend_states[friend_i].abs_pos,
        #       friend_states[friend_i].vel, friend_states[friend_i].is_dead)

    # print("observed state: ", state)
    # print(self_state.pos_x, self_state.pos_y, self_state.vel_x, self_state.vel_y)

    return self_state, adv_states, friend_states


def if_in_box(agent_state):
    abs_x = agent_state.abs_pos[0]
    abs_y = agent_state.abs_pos[1]
    if abs(abs_x) > 1 or abs(abs_y) > 1:
        # print("Out of box")
        return False
    else:
        # print("In box")
        return True


def aggressive(self_state, adv_states, num_adversaries, dist_max):
    # if self is dead, keep still
    if self_state.is_dead:
        action = 0
        return action

    # if self is not dead, but all advs are either dead or out of box
    are_all_adv_dead_out = True
    for adv_i in range(num_adversaries):
        if (not adv_states[adv_i].is_dead) and (if_in_box(adv_states[adv_i])):
            are_all_adv_dead_out = False
    if are_all_adv_dead_out:
        action = 0
        return action

    # if self is not dead and there are advs alive, find the nearest one
    dists = []
    nearest_dist = dist_max
    nearest_adv_id = -1
    for dist_i in range(num_adversaries):
        # print("adv: ", dist_i, "state: ", adv_states[dist_i].abs_pos)
        # adv should be alive and in the bounding box
        if (not adv_states[dist_i].is_dead) and (if_in_box(adv_states[dist_i])):
            delta_pos = np.array([adv_states[dist_i].rel_pos[0], adv_states[dist_i].rel_pos[1]])
            dists.append(np.sqrt(np.sum(np.square(delta_pos))))
            # print("dist_i: ", dist_i)
            # print("dists: ", dists)
            # print(delta_pos, dists[dist_i])
            if nearest_dist > dists[dist_i]:
                nearest_dist = dists[dist_i]
                nearest_adv_id = dist_i
                assert nearest_dist < dist_max, 'invalid nearest_dist'
        else:
            dists.append(dist_max)

    # evaluate the relative pos of the nearest live adv
    delta_pos_nearest_adv = np.array([adv_states[nearest_adv_id].rel_pos[0],
                                      adv_states[nearest_adv_id].rel_pos[1]])
    # print(nearest_adv_id, np.sqrt(np.sum(np.square(delta_pos_nearest_adv))), nearest_dist)
    #assert np.sqrt(np.sum(np.square(delta_pos_nearest_adv))) == nearest_dist, 'unmatched nearest distance'

    if delta_pos_nearest_adv[0] == 0 and delta_pos_nearest_adv[1] == 0:
        action = 0
    elif abs(delta_pos_nearest_adv[0]) >= abs(delta_pos_nearest_adv[1]):
        if delta_pos_nearest_adv[0] > 0:
            action = 1
        else:
            action = 2
    else:
        if delta_pos_nearest_adv[1] > 0:
            action = 3
        else:
            action = 4

    return action


def defensive(self_state, adv_states, num_adversaries, dist_max):
    # if self is dead, keep still
    if self_state.is_dead:
        action = 0
        return action

    # if self is not dead, but all advs are either dead or out of box
    are_all_adv_dead_out = True
    for adv_i in range(num_adversaries):
        if (not adv_states[adv_i].is_dead) and (if_in_box(adv_states[adv_i])):
            are_all_adv_dead_out = False
    if are_all_adv_dead_out:
        action = 0
        return action

    # if self is not dead and there are advs alive and attacking, move towards the safest angle