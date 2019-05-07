import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import math
import copy

attack_angle = 90
defense_angle = 90
fire_range = 0.5


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 3
        num_adversaries = 3
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 0

        self.num_green = copy.deepcopy(num_good_agents)
        self.num_red = copy.deepcopy(num_adversaries)
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.04 if agent.adversary else 0.04
            #agent.accel = 3.0 if agent.adversary else 2.0
            agent.accel = 2.0 if agent.adversary else 2.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            #agent.max_speed = 1.5 if agent.adversary else 1.2
            agent.max_speed = 1.0 if agent.adversary else 1.0
            #agent.max_speed = 1.0 if agent.adversary else 0.0  ###changed by liyuan
            agent.death = False

            agent.chi = np.array([0.1,0])

            if agent.adversary:
                agent.lock_num=[0 for j in range(num_good_agents)]
            else:
                agent.lock_num=[0 for j in range(num_adversaries)]
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.death = False
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
        
        if agent.adversary:
                agent.lock_num=[0 for j in range(self.num_green)]
        else:
                agent.lock_num=[0 for j in range(self.num_red)]


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent) and a.death == False:
                    collisions += 1
            return collisions
        else:
            return 0

    '''
    def is_collision(self, agent1, agent2):
        if agent1.death or agent2.death:
            return False
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        #dist_min = agent1.size + agent2.size
        dist_min = 0.1
        return True if dist < dist_min else False
    '''

    ##liyuan: compute the number of lokcing number of the agent
    def compute_lock_num(self, agent, world):
        opponent = []
        if agent.adversary:
            opponent = self.good_agents(world)
        else:
            opponent = self.adversaries(world)
   
        for i, opp in enumerate(opponent):
            if self.is_collision(opp,agent):
                agent.lock_num[i] += 1
            else:
                agent.lock_num[i] = 0
    
    ###liyuan: True if agent1 win, False for others
    def is_collision(self, agent1, agent2):
        if agent1.death or agent2.death:
            return False

        ###liyuan:judged by angle
        delta_pos = agent2.state.p_pos - agent1.state.p_pos
        distance = np.sqrt(np.sum(np.square(delta_pos)))
        if distance <= 0.0001:
            return False
        
        agent1_chi = [agent1.state.p_vel[0],agent1.state.p_vel[1]]
        
        if agent1.state.p_vel[0] < 1e-5 and agent1.state.p_vel[1]<1e-5:
            agent1_chi[0] = 0.1
            agent1_chi[1] = 0
        agent2_chi = [agent2.state.p_vel[0],agent2.state.p_vel[1]]
       
        if agent2.state.p_vel[0] < 1e-5 and agent2.state.p_vel[1]<1e-5:
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

        if distance < fire_range and agent2_angle*180/math.pi>revised_defense and agent1_angle*180/math.pi<attack_angle/2:
            return True
        #elif distance < fire_range and agent2_angle*180/math.pi<attack_angle/2 and agent1_angle*180/math.pi>revised_defense:
            #return True,2
        #else:
        return False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        ####added by liyuan
        if agent.death == True:
            return 0
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        #shape = False
        shape = True
        adversaries = self.adversaries(world)
        '''
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                ###changed by liyuan
                if adv.death == True:
                    continue
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        '''
        self.compute_lock_num(agent, world)
        if agent.collide:
            for i,a in enumerate(adversaries):
                ###changed by liyuan
                #if self.is_collision(a, agent) and a.death == False:
                if agent.lock_num[i]>=3 and a.death == False:
                    #rew -= 10
                    agent.death = True
        

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)
        
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            if (x > 1.0):
                rew -= 20
                break

        return rew

    def adversary_reward(self, agent, world):
        ####added by liyuan
        if agent.death == True:
            return 0
        # Adversaries are rewarded for collisions with agents
        rew = 0
        #shape = False
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        
        '''
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                ###rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
                if adv.death == False:
                    dis = []
                    for a in agents:
                        if a.death == False:
                            dis.append(np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))))
                    if len(dis) > 0:
                        rew -= 0.1 * min(dis)
        '''
        if shape: 
            dis = []
            for a in agents:
                if a.death == False:
                    dis.append(np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))))
            if len(dis) > 0:
                rew -= 0.1*min(dis)
        
        self.compute_lock_num(agent, world)
        if agent.collide:
            for ag in agents:
                for i,adv in enumerate(adversaries):
                    ###changed by liyuan
                    #if self.is_collision(adv,ag) and ag.death == False and adv.death == False:
                    if ag.lock_num[i]>=3 and ag.death == False and adv.death == False:
                        if adv is agent:
                            rew += 40
                        else:
                            rew += 20
                        break
        
        ###if the red agent is eatten
        if agent.collide:
            for i,ag in enumerate(agents):
                #if self.is_collision(ag, agent) and ag.death == False:
                if ag.death == False and agent.lock_num[i]>=3:
                    rew -= 40
                    agent.death = True
                    break
        
        for adv in adversaries:
            if adv.death == False:
                exceed = False
                for p in range(world.dim_p):
                    x = abs(adv.state.p_pos[p])
                    if (x > 1.0):
                        exceed = True
                        break
                if exceed == True:
                    if adv is agent:
                        rew -= 40
                    else:
                        rew -=20
                    break

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        other_chi = []
        our_chi = []

        my_chi = np.zeros(1)
        if agent.state.p_vel[0]<1e-5 and agent.state.p_vel[1]<1e-5:
            my_chi[0] = 0
        elif agent.state.p_vel[0]<1e-5:
            my_chi[0] = math.pi/2
        else:
            my_chi[0] = math.atan(agent.state.p_vel[1]/agent.state.p_vel[0])        
        our_chi.append(my_chi)

        for other in world.agents:
            if other is agent: continue            
            ###changed by liyuan
            if other.death:
                comm.append(np.zeros(world.dim_c))
                other_pos.append(np.zeros(world.dim_p))
                other_vel.append(np.zeros(world.dim_p))
                tmp_chi = np.zeros(1)
                other_chi.append(tmp_chi)
            else:
                comm.append(other.state.c)
                other_pos.append(other.state.p_pos - agent.state.p_pos)
                #if not other.adversary:
                other_vel.append(other.state.p_vel)

                tmp_chi = np.zeros(1)
                if other.state.p_vel[0]<1e-5 and other.state.p_vel[1]<1e-5:
                    tmp_chi[0] = 0
                elif other.state.p_vel[0]<1e-5:
                    tmp_chi[0] = math.pi/2
                else:
                    tmp_chi[0] = math.atan(other.state.p_vel[1]/other.state.p_vel[0])
                other_chi.append(tmp_chi)

            #comm.append(other.state.c)
            #other_pos.append(other.state.p_pos - agent.state.p_pos)
            #if not other.adversary:
                #other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel + our_chi + other_chi)

    ##added by liyuan: if all green nodes die, this epsoid is over.
    def done(self, agent, world):
        allDie = True
        agents = self.good_agents(world)
        for agent in agents:
            if agent.death == False:
                allDie = False
                break
        return allDie
