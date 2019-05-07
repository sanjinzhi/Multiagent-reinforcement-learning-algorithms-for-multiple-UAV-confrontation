import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import math
import copy

attack_angle = 90
defense_angle = 90
fire_range = 0.3
first_level = fire_range+0.3
second_level = fire_range+0.1

class RuleAgent:
    def __init__(self,env):
        self.env = env
    
    def target_assignment(self,target_allocation):
        new_allocation=[[] for i in range(self.env.n)]
        allocation_distance=[[-1 for j in range(self.env.n)] for i in range(self.env.n)]
        used_agent=[-1 for i in range(self.env.n)]

        for i,ag in enumerate(self.env.agents):
            if not ag.adversary and ag.death == False:
                for i_index in target_allocation[i]:
                    if self.env.agents[i_index].death == False:
                        used_agent[i_index] = i
                        delta_pos=ag.state.p_pos - self.env.agents[i_index].state.p_pos
                        distance = np.sqrt(np.sum(np.square(delta_pos)))
                        allocation_distance[i_index][i] = distance
                        new_allocation[i].append(i_index)
        
        
        for i,ag in enumerate(self.env.agents):
            if not ag.adversary and ag.death == False:
                for j,adv in enumerate(self.env.agents):
                    if adv.adversary and adv.death == False:
                        delta_pos=ag.state.p_pos - adv.state.p_pos
                        distance = np.sqrt(np.sum(np.square(delta_pos)))
                        if distance <= second_level:
                            if used_agent[j] > -1 and allocation_distance[j][used_agent[j]]>distance:
                                if j in new_allocation[used_agent[j]]:
                                    new_allocation[used_agent[j]].remove(j)                                
                                new_allocation[i].append(j)
                                used_agent[j] = i
                                allocation_distance[j][i]=distance
                            elif used_agent[j]<=-1:
                                new_allocation[i].append(j)
                                used_agent[j] = i
                                allocation_distance[j][i]=distance
                        elif distance < first_level:
                            if used_agent[j]<=-1:
                                new_allocation[i].append(j)
                                used_agent[j] = i
                                allocation_distance[j][i]=distance

        
        red_for_green=[[] for i in range(self.env.n)]
        for i,ag in enumerate(self.env.agents):
            if not ag.adversary and ag.death == False:
                for j,adv in enumerate(self.env.agents):
                    if adv.adversary and adv.death == False:
                        delta_pos=ag.state.p_pos - adv.state.p_pos
                        distance = np.sqrt(np.sum(np.square(delta_pos)))
                        if distance <= second_level:
                            red_for_green[i].append(j)
        for i,ag in enumerate(self.env.agents):
            if not ag.adversary and ag.death == False:
                if len(red_for_green[i]) > 1:
                    for j in red_for_green[i]:
                        if j in new_allocation[used_agent[j]]:
                            new_allocation[used_agent[j]].remove(j)
                        new_allocation[i].append(j)
                        used_agent[j] = i               
        
        
        green_red_num=[[0,0] for i in range(self.env.n)]
        
        for j,adv in enumerate(self.env.agents):
            if adv.adversary and adv.death == False:
                for i,ag in enumerate(self.env.agents):
                    if not ag.adversary and ag.death == False:
                        delta_pos=ag.state.p_pos - adv.state.p_pos
                        distance = np.sqrt(np.sum(np.square(delta_pos)))
                        if distance <= first_level:
                            green_red_num[j][0] += 1 
                for i,ag in enumerate(self.env.agents):
                    if ag.adversary and ag.death == False:
                        delta_pos=ag.state.p_pos - adv.state.p_pos
                        distance = np.sqrt(np.sum(np.square(delta_pos)))
                        if distance <= first_level:
                            green_red_num[j][1] += 1

        nearest_friend = [-1 for i in range(self.env.n)]
        
        for i,ag in enumerate(self.env.agents):
            if ag.adversary and ag.death == False:
                min_dis = 100
                min_index = -1
                for j,adv in enumerate(self.env.agents):
                    if adv.adversary and adv.death == False and adv is not ag:
                        if green_red_num[j][0]<=green_red_num[j][1]:
                            delta_pos=ag.state.p_pos - adv.state.p_pos
                            distance = np.sqrt(np.sum(np.square(delta_pos)))
                            if distance<min_dis:
                                min_dis = distance
                                min_index = j
                nearest_friend[i] = min_index      


        return  new_allocation,green_red_num,nearest_friend


    def first_rule_action(self,agent,green_red_num,nearest_friend,target_allocation):
        ####escape when the number of green nodes is more than the number of red nodes
        for j,adv in enumerate(self.env.agents):
            if adv.adversary and adv.death == False and adv is agent:
                if  green_red_num[j][0]>green_red_num[j][1] and nearest_friend[j]>-1:
                    delta_pos=self.env.agents[nearest_friend[j]].state.p_pos - agent.state.p_pos
                    actions=self.action_number(delta_pos)
                    return actions
        ##attack
        ag_index = -1
        for i,ag in enumerate(self.env.agents):
            if not ag.adversary and ag.death == False:
                for j in target_allocation[i]:
                    if agent is self.env.agents[j]:
                        ag_index = i
                        break
        
        actions=np.array([])
        if ag_index > -1:
             delta_pos=self.env.agents[ag_index].state.p_pos - agent.state.p_pos
             actions=self.action_number(delta_pos)
        return actions
    
    def action_number(self,min_delta):
        if abs(min_delta[0])<1e-5 and abs(min_delta[1])<1e-5:
                action = 0
        if abs(min_delta[0]) >= abs(min_delta[1]):
            if min_delta[0]>0:
                action = 1
            else:
                action = 2
        else:
            if min_delta[1]>0:
                action = 3
            else:
                action = 4
        action_space=np.array([0 for i in range(5)])
        action_space[action]=1
        return action_space
        


    
    def rule_action(self,agent):        
        min_dis = 100
        min_index = -1
        min_delta=[]
        for i,ag in enumerate(self.env.agents):
            if ag.adversary != agent.adversary:
                delta_pos=ag.state.p_pos - agent.state.p_pos
                distance = np.sqrt(np.sum(np.square(delta_pos)))
                if distance < min_dis:
                    min_dis = copy.deepcopy(distance)
                    min_index = copy.deepcopy(i)
                    min_delta = copy.deepcopy(delta_pos)

        if min_dis < first_level:
            if abs(min_delta[0])<1e-5 and abs(min_delta[1])<1e-5:
                action = 0
            if abs(min_delta[0]) >= abs(min_delta[1]):
                if min_delta[0]>0:
                    action = 1
                else:
                    action = 2
            else:
                if min_delta[1]>0:
                    action = 3
                else:
                    action = 4
            action_space=np.array([0 for i in range(5)])
            action_space[action]=1
            return action_space
        else:
            return np.array([])
        

    
    def will_hit(self, agent1, agent2,hit_range):
        if agent1.death or agent2.death:
            return False

        ###liyuan:judged by angle
        delta_pos = agent2.state.p_pos - agent1.state.p_pos
        distance = np.sqrt(np.sum(np.square(delta_pos)))
        if distance <= 1e-5:
            return False
        
        agent1_chi = [agent1.state.p_vel[0],agent1.state.p_vel[1]]

        if abs(agent1.state.p_vel[0]) < 1e-5 and abs(agent1.state.p_vel[1])<1e-5:
            agent1_chi[0] = 0.1
            agent1_chi[1] = 0
        agent2_chi = [agent2.state.p_vel[0],agent2.state.p_vel[1]]

        if abs(agent2.state.p_vel[0]) < 1e-5 and abs(agent2.state.p_vel[1])<1e-5:
            agent2_chi[0] = 0.1
            agent2_chi[1] = 0

        agent1_chi_value = np.sqrt(np.sum(np.square(agent1_chi)))
        agent1_cross = (delta_pos[0]*agent1_chi[0]+delta_pos[1]*agent1_chi[1])/(distance*agent1_chi_value)
        if agent1_cross < -1:
           agent1_cross  = -1
        if agent1_cross > 1:
           agent1_cross = 1
        agent1_angle = math.acos(agent1_cross)


        agent2_chi_value = np.sqrt(np.sum(np.square(agent2_chi)))
        agent2_cross = (-delta_pos[0]*agent2_chi[0]-delta_pos[1]*agent2_chi[1])/(distance*agent2_chi_value)
        if agent2_cross < -1:
           agent2_cross  = -1
        if agent2_cross > 1:
           agent2_cross = 1
        agent2_angle = math.acos(agent2_cross)

        revised_defense = 180-defense_angle/2
        if distance < hit_range and agent2_angle*180/math.pi>revised_defense and agent1_angle*180/math.pi<attack_angle/2:
            return True
        #elif distance < hit_range and agent2_angle*180/math.pi<attack_angle/2 and agent1_angle*180/math.pi>revised_defense:
            #return True,2
        #else:
        return False
