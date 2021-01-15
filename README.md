# Optimizing-UAV-trajectory-for-maximum-data-rate-via-Q-Learning
1. ```Goal```    
    The goal of our subject is to optimize the UAV trajectory for maximum data rate via Q-Learning in avoidance of obstacles.   

2. ```Technical approaches```    
    We created an environment for UAV, unmanned aerial vehicle in Unreal Engine. We applied Unreal Engine Blueprint to make random environment whenever the program starts. AirSim API was used to control the simulated UAV on Unreal Engine. Python programming language was used because AirSim API works on C++ or Python.
    
    We have 7x7 grids. Each grid is 100m long. The grids are the states to which UAV has to fly in accordance with the certain action. We divided 3 levels for different altitudes. UAV cannot fly beyond 200 m and below 50 m. UAV can perform six actions –forward, backward, left, right, top and bottom. Buildings are 60 m, 150 m, 250 m high respectively. We put 5 routers in the environment. The positions of buildings and routers are random.  
    
3.```Research and Analysis``` 
   Our Project is simulation-based optimization and hence Reinforcement learning (RL) is the best approach. Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment in order to maximize some notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning. To optimize the UAV path, Q-Learning algorithm was used. Q-learning uses future rewards to influence the current action given a state and therefore helps the agent select best actions that maximize total reward.  
   We have 5 routers. It means that there are 5 goals that UAV has to fly. We trained 5000 episodes for each goal. Therefore, we trained 25,000 episodes for the whole program. According to Q-Learning algorithm, we initialize the Q-table, choose an action, perform action, measure reward and update Q-table after one episode is finished. We initialized Q-table with all 0s. 
   We defined our reward matrix. If UAV hits an obstacle, we give -10 reward. UAV gets data rate if it reaches the goal. UAV possesses -1 if collision does not happen or it does not reach the goal.  A2G , Air To Ground channel is defined by LoS, Line of Sight and NLoS, Non-Line of sight links between UAV and user. Data Rate is calculated by using path loss, recieved power, noise (-90 dBm) and bandwidth (2 MHz). We finally gained the reward matrix. There are 49 states in our UAV environment. 0 is the starting point. Our altitude is 50 m (level 0).  We randomly go to 1. We randomly choose action Forward 0. There is no obstacle nor goal in state 1 so the reward will be -1. In state 1, our future action will be one of 6 actions [ Forward 0, Backward 1, Left 2, Right 3, Up 4 , Down 5] 
So our Q function for this state is 
Q( 0, 0, 0 ) = Q( 0, 0, 0 ) + 0.01 x [ R( 0, 0, 0 )  + 0.9 x [max (Q( 1, 0, 0 ), Q( 1, 0, 1 ), 
Q( 1, 0, 2 ), Q( 1, 0, 3), Q( 1, 0, 4 ), Q( 1, 0, 5 ) )- Q( 0, 0, 0 ) ]   = 0 + 0.01 x [ -1 + 0.9 x [0 – 0] ]   = -0.01   
 
In training case, Q-Matrix is updated after each episode is finished.  By this way, Q-table is updated 5000 times for one goal. After the training case is finished (all goals are trained), the testing case begins. In testing case, UAV flies to reach each goal according to the maximum Q-Matrix value for that state and action. By this way, our UAV flies autonomously with optimal path to all the routers.  
 
4. ```Conclusion```
 
 Our project optimizes the UAV trajectory for any environment by learning itself. We can apply our program in hazardous environment where humans cannot go such as after an earthquake.  



