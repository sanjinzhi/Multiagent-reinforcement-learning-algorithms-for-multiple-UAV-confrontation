# multiagent-confrontation
This is the source code of "Efficient training techniques for multi-agent reinforcement learning in combatant tasks", 
we construct a multi-agent confrontation environment originated from a combatant scenario of multiple unman aerial vehicles. 
To begin with, we consider to solve this confrontation problem with two types of MARL algorithms. 
One is extended from the classical deep Q-network for multi-agent settings (MADQN). 
The other one is extended from the state-of-art multi-agent reinforcement method, multi-agent deep deterministic policy gradient (MADDPG). 
We compare the two methods for the initial confrontation scenario and find that MADDPG outperforms MADQN. 
Then with MADDPG as the baseline, we propose three efficient training techniques, i.e., scenario-transfer training, self-play training and rule-coupled training.
