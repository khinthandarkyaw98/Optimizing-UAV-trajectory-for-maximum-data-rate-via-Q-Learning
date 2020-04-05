# Optimizing-UAV-trajectory-for-maximum-data-rate-via-Q-Learning
1. Goal    
    The goal of our subject is to optimize the UAV trajectory for maximum data rate via Q-Learning in avoidance of obstacles.   

2. Technical approaches    
    We created an environment for UAV, unmanned aerial vehicle in Unreal Engine. We applied Unreal Engine Blueprint to make random environment whenever the program starts. AirSim API was used to control the simulated UAV on Unreal Engine. Python programming language was used because AirSim API works on C++ or Python.
    We have 7x7 grids. Each grid is 100m long. The grids are the states to which UAV has to fly in accordance with the certain action. We divided 3 levels for different altitudes. UAV cannot fly beyond 200 m and below 50 m. UAV can perform six actions –forward, backward, left, right, top and bottom. Buildings are 60 m, 150 m, 250 m high respectively. We put 5 routers in the environment. The positions of buildings and routers are random.  
    
3.	Research and Analysis 
   Our Project is simulation-based optimization and hence Reinforcement learning (RL) is the best approach. Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment in order to maximize some notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning. To optimize the UAV path, Q-Learning algorithm was used. Q-learning uses future rewards to influence the current action given a state and therefore helps the agent select best actions that maximize total reward.  

