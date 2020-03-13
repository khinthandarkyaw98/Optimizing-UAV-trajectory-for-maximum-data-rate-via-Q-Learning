import numpy as np
from numpy import *
import time
import airsim
from sqnr_auxfuncs_v2 import *
import logging
import matplotlib.pyplot as plt

logging.basicConfig(filename='sqnr_training_v2.log',level=logging.DEBUG)

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

gamma = 0.9 #0.9
alpha = 0.01 #0.01
num_states = 49
num_actions = 6
num_altitudes = 3
num_steps_in_episode = 100
num_episodes=5000
random_state=np.random.RandomState(2)
random_seed = np.random.RandomState(10)

rewards_all_episodes=[] #__TBA

epsilon = 1 #__TBA
epsilon_max=1 #__TBA
epsilon_min=0.01 #__TBA
epsilon_decay_rate=0.9993 

buildings_list, bld_heights = get_buildings_list(client)
goals_list, goals_poses=get_routers_list(client)

print("buildings list: %s"%buildings_list)
logging.debug("buildings list: %s"%buildings_list)
print("buildings heights: %s"%bld_heights)
logging.debug("buildings heights: %s"%bld_heights)
print("goals list: %s"%goals_list)
logging.debug("goals list: %s"%goals_list)
print("goals poses: %s"%goals_poses)
logging.debug("goals poses: %s"%goals_poses)

rewards,q_matrix = [],[]

for i in range(len(goals_list)):
	reward, qmatrix = init_reward_q(num_altitudes, num_states, num_actions, goals_list[i], goals_poses[i], buildings_list, bld_heights)
	rewards.append(reward)
	q_matrix.append(qmatrix)
save('sqnr_total_rewards.npy',rewards)
save('sqnr_total_initial_q.npy',q_matrix)	

prev_state=0
prev_altitude=0
for i in range(len(goals_list)):
	epsilon, episode=1,1
	rewards_all_episodes=[] 
	goal=goals_list[i]
	init_state=prev_state
	init_altitude=prev_altitude
	while episode<num_episodes:
		step=1
		current_state=0
		current_altitude=0
		next_state=-1
		next_altitude=-1
		done=False
		rewards_current_episode=0
		
		while done==False:
			print("step: %s"%step)
			logging.debug("step:%s"%step)
			
			print("current state:%s"%current_state)
			logging.debug("current state:%s"%current_state)
			
			print("current altitude:%s"%current_altitude)
			logging.debug("current altitude:%s"%current_altitude)
			
			if current_state==goal:
				print("GOAL reached------------------")
				logging.debug("GOAL reached------------------")
				prev_altitude=current_altitude
				prev_state=current_state
				done=True
				break
				
			explore_exploit_n=random.uniform(0,1)
			if explore_exploit_n>epsilon:
				max_q_value = max(q_matrix[i][current_altitude, current_state, :])
				results = np.where(q_matrix[i][current_altitude,current_state]==max_q_value)
				results=results[0]
				if results.size>1: 
					random_seed.shuffle(results) 
				action = results[0] 
			else:
				actions = np.array(list(range(num_actions)))
				random_state.shuffle(actions)
				action = actions[0]
		
			print("action: %s"%action)
			logging.debug('action: %s'%action)
	
			next_state=get_next_state(action, current_state)
			print("next state: %s"%next_state)
			logging.debug('next state: %s'%next_state)
		
			next_altitude = get_next_altitude(action, current_altitude)
			print("next altitude: %s"%next_altitude)
			logging.debug("next altitude: %s"%next_altitude)
			
			reward = update_q_value(q_matrix[i], rewards[i], current_state, next_state, current_altitude, next_altitude, action, alpha, gamma)
			rewards_current_episode=+reward
			
			if next_state in buildings_list:
				index = buildings_list.index(next_state)
				if next_altitude<=bld_heights[index]:
					print("collision occurs-----------")
					logging.debug("Collision----------")
			else:
				current_state=next_state
				current_altitude=next_altitude
			
			if step>num_steps_in_episode: done=True
			print('EOS----------------------')
			logging.debug('EOS----------------------')
			step=step+1
		print("Episode %s done---------------"%episode)
		logging.debug("Episode %s done--------------"%episode)
		
		if epsilon>=epsilon_min:
			epsilon*=epsilon_decay_rate
		
		rewards_all_episodes.append(rewards_current_episode)
		episode=episode+1
	save('q_matrix_'+str(goal)+'.npy',q_matrix[i])
	#plot rewards vs episodes graph
	plt.title('Rewards vs. Episodes')
	plt.plot(range(len(rewards_all_episodes)),rewards_all_episodes)
	plt.xlabel('episodes')
	plt.ylabel('rewards')
	plt.show()

client.takeoffAsync().join()
start_states=[]
start_states.append(0)
for i in range(len(goals_list)-1):
	start_states.append(goals_list[i])
start_altitude=0
for i in range(len(goals_list)):
	start_altitude=get_optimal_route(rewards[i],q_matrix[i],goals_list[i],start_states[i],start_altitude,client)

client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
