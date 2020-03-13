import numpy as np
from numpy import *
import time
import airsim
import logging
import matplotlib.pyplot as plt

logging.basicConfig(filename='sqnr_testing_v2.log',level=logging.DEBUG)

def get_buildings_list(client):
	buildings=[]
	bld_heights = []
	i=0
	state=-1
	while i<=1:
		pose = client.simGetObjectPose('bld1_C_'+str(i))
		print("Building pose: ")
		print(pose)
		if not math.isnan(pose.position.x_val):
			altitude=0
			bld_heights.append(altitude)
			state = (int(math.ceil(abs(pose.position.y_val))/10)*7)+(int(pose.position.x_val/10))
			print(state)
			buildings.append(state)
		i=i+1
	i=0
	while i<=11:
		pose = client.simGetObjectPose('bld2_C_'+str(i))
		print("Building pose: ")
		print(pose)
		if not math.isnan(pose.position.x_val):
			altitude=1
			bld_heights.append(altitude)
			state = (int(math.ceil(abs(pose.position.y_val))/10)*7)+(int(pose.position.x_val/10))
			print(state)
			buildings.append(state)
		i=i+1
	i=0
	while i<=1:
		pose = client.simGetObjectPose('bld3_C_'+str(i))
		print("Building pose: ")
		print(pose)
		if not math.isnan(pose.position.x_val):
			altitude=2
			bld_heights.append(altitude)
			state = (int(math.ceil(abs(pose.position.y_val))/10)*7)+(int(pose.position.x_val/10))
			print(state)
			buildings.append(state)
		i=i+1
	return buildings, bld_heights

def get_routers_list(client):
	routers,routers_poses=[],[]
	i=0
	while i<=4:
		pose_xy=[]
		pose = client.simGetObjectPose('node_C_'+str(i))
		print("Goal pose: ")
		print(pose)
		if not math.isnan(pose.position.x_val):
			pose_xy.append(pose.position.x_val)
			pose_xy.append(pose.position.y_val)
			if pose.position.x_val%10>5:
				x_val=int(int(math.ceil(pose.position.x_val/10.0))*10)/10
			else: x_val=int(pose.position.x_val/10)
			state = int((int(math.ceil(abs(pose.position.y_val))/10)*7)+x_val)
			routers.append(state)
			routers_poses.append(pose_xy)
		i=i+1
	return routers,routers_poses

def init_reward_q(num_altitudes, num_states, num_actions, goal, goal_pose, buildings_list, bld_heights): #__TBA
	reward_matrix = np.zeros((num_altitudes,num_states, num_actions))
	q_matrix = np.zeros_like(reward_matrix)
	
	reward_matrix[:,:,:]=-1	
	reward_matrix[:,0:7,3]=-10
	reward_matrix[:,42:49,2]=-10
	for i in range(num_states):
		if i%7==0: reward_matrix[:,i,1]=-10
		if i%7==6: reward_matrix[:,i,0]=-10	
	reward_matrix[0,:,5]=-5
	reward_matrix[2,:,4]=-5
	   
	for i in range(num_altitudes):  #___TBA
		if goal not in list(range(6,55,7)): 
			reward_matrix[i,goal+1,1]=calculate_reward(i,goal,goal_pose) #__TBA
		if goal not in list(range(7)): 
			reward_matrix[i,goal-7,2]=calculate_reward(i,goal,goal_pose) #___TBA
		if goal not in list(range(42,49)): 
			reward_matrix[i,goal+7,3]=calculate_reward(i,goal,goal_pose) #___TBA
		if goal not in list(range(0,49,7)): 
			reward_matrix[i,goal-1,0]=calculate_reward(i,goal,goal_pose) #___TBA
		reward_matrix[1,goal,5]=calculate_reward(0,goal,goal_pose)
		reward_matrix[2,goal,5]=0.6*calculate_reward(1,goal,goal_pose)
		reward_matrix[0,goal,4]=0.6*calculate_reward(1,goal,goal_pose)
		reward_matrix[1,goal,4]=0.3*calculate_reward(2,goal,goal_pose)

	
	for i in range(len(buildings_list)):
		building=buildings_list[i]
		height=bld_heights[i]
		if building not in list(range(6,55,7)): 
			for j in range(height+1):
				reward_matrix[j,building+1,1]=-10
		if building not in list(range(7)): 
			for j in range(height+1):
				reward_matrix[j,building-7,2]=-10
		if building not in list(range(42,49)): 
			for j in range(height+1):
				reward_matrix[j,building+7,3]=-10
		if building not in list(range(0,49,7)): 
			for j in range(height+1):
				reward_matrix[j,building-1,0]=-10
		for j in range(height+1):
			reward_matrix[j,building,:]=0
	
	save('sqnr_reward_matrix_'+str(goal)+"_"+str(time.time())+'.npy', reward_matrix)#___TBA
	print('reward matrix saved.') #___TBA
	save('sqnr_inital_q_matrix_'+str(goal)+'.npy',q_matrix) #___TBA
	
	return reward_matrix, q_matrix

def calculate_reward(altitude, goal_state,goal_pose): #__TBA
	n_los,n_nlos = 1.0, 20.0
	a=9.61
	b=0.16
	theta, opposite = calculate_theta(altitude,goal_state,goal_pose) #__TBA
	carrier_freq = 2412000000  #Hz 
	transmit_p = 5 #W
	bandwidth = 2000000 #Hz
	noise = -90 #dBm 
	if theta==0:
		hypotenuse=opposite
	else: 
		hypotenuse = opposite/math.sin(theta) # _TBC
	c = 299792458
	prob_los = 1/(1+a*math.exp(-b*((180/math.pi)*theta-a)))
	path_loss = 20*math.log(((4*math.pi*carrier_freq*hypotenuse)/c),10)+ prob_los*n_los + (1-prob_los)*n_nlos

	path_loss = 10**(-path_loss/10) #path_loss and noise in watt
	noise = 10**(noise/10)

	received_p = transmit_p - path_loss
	data_rate = bandwidth*math.log((1+received_p/noise),2)
	data_rate=data_rate/1000
	return data_rate
	
def calculate_theta(altitude, goal_state, goal_pose): #__TBA
	ue_x_val=goal_pose[0]
	ue_y_val=abs(goal_pose[1])
	
	uav_x_val = (goal_state%7)*10 #__TBA
	uav_y_val = (goal_state/7)*10 #__TBA
	
	if altitude==0:
		z_val = 50
	elif altitude==1:
		z_val=125
	elif altitude==2:
		z_val=200
	opposite = z_val
	adjacent = math.sqrt(((abs(uav_x_val-ue_x_val))*10)**2+((abs(uav_y_val-ue_y_val))*10)**2) #__TBA
	if adjacent==0: theta = 0
	else: theta = math.atan(opposite/adjacent) 
	return theta, opposite


def update_q_value(q_matrix, reward_matrix, current_state, next_state, current_altitude, next_altitude, action, alpha, gamma):
	reward = reward_matrix[current_altitude,current_state, action]
	new_q = q_matrix[current_altitude, current_state, action]+ alpha * (reward +(gamma*max(q_matrix[next_altitude, next_state, :]))- q_matrix[current_altitude, current_state, action])
	q_matrix[current_altitude, current_state, action]=new_q
	return reward
	
def get_next_altitude(action, current_altitude):
	next_altitude = current_altitude
	if action==4 and current_altitude==0: next_altitude=1
	elif action==4 and (current_altitude==1 or current_altitude==2): next_altitude=2
	if action==5 and (current_altitude==0 or current_altitude==1): next_altitude=0
	elif action==5 and current_altitude==2: next_altitude=1
	return next_altitude
	
def get_next_state(action, current_state):
	next_state=current_state
	if action==0 and current_state%7!=6: next_state=current_state+1
	elif action==1 and current_state%7!=0: next_state=current_state-1
	elif action==2 and (current_state not in list(range(42,49))): next_state=current_state+7
	elif action==3 and (current_state not in list(range(7))) : next_state=current_state-7
	return next_state

def uav_take_action(action, client, current_altitude):
	done=False
	pose = client.simGetVehiclePose()
	if action==0: 		
		client.moveToPositionAsync(pose.position.x_val+10, pose.position.y_val, pose.position.z_val, 3).join()
		client.hoverAsync().join()
	elif action==1: 				
		client.moveToPositionAsync(pose.position.x_val-10, pose.position.y_val, pose.position.z_val, 3).join()
		client.hoverAsync().join()
	elif action==2:
		client.moveToPositionAsync(pose.position.x_val, pose.position.y_val-10, pose.position.z_val, 3).join()
		client.hoverAsync().join()
	elif action==3:
		client.moveToPositionAsync(pose.position.x_val, pose.position.y_val+10, pose.position.z_val, 3).join()
		client.hoverAsync().join()
	elif action==4 and current_altitude<2 and current_altitude>=0: 
		client.moveToPositionAsync(pose.position.x_val, pose.position.y_val, pose.position.z_val-7.5, 3).join() #75 meters
		client.hoverAsync().join()
	elif action==5 and current_altitude>0 and current_altitude<=2:
		client.moveToPositionAsync(pose.position.x_val, pose.position.y_val, pose.position.z_val+7.5, 4).join()
		client.hoverAsync().join()
	
	if client.simGetCollisionInfo().has_collided==True:
		done=True

def get_optimal_route(reward_matrix, q_matrix,goal,start_state,start_altitude, client):
	current_state = start_state
	current_altitude=start_altitude
	next_altitude=-1
	next_state=-1
	done=False
	z_val = client.simGetVehiclePose().position.z_val

	if current_altitude==0:
		client.moveToZAsync(-5,3).join()
	elif current_altitude==1:
		client.moveToZAsync(-12.5,3).join()
	elif current_altitude==2:
		client.moveToZAsync(-20,3).join()

	while done==False:
		print("current state:%s"%current_state)
		logging.debug("current state:%s"%current_state)
			
		print("current altitude:%s"%current_altitude)
		logging.debug("current altitude:%s"%current_altitude)

		if current_state==goal:
			print("GOAL reached----------------")
			logging.debug("GOAL reached------------")
			done=True
			break
		else:
			action = argmax(q_matrix[current_altitude, current_state])
		print("action: %s"%action)
		logging.debug('action: %s'%action)
		
		uav_take_action(action,client,current_altitude)	

		next_state=get_next_state(action, current_state)
		print("next state: %s"%next_state)
		logging.debug('next state: %s'%next_state)
		
		next_altitude = get_next_altitude(action, current_altitude)
		print("next altitude: %s"%next_altitude)
		logging.debug("next altitude: %s"%next_altitude)
		
		if next_state==goal:
			data_rate=reward_matrix[current_altitude,current_state,action]/1000
			print("Data Rate: %s Mbps"%data_rate)

		print('-----------------------------------------------------')
		current_state=next_state
		current_altitude=next_altitude
	return current_altitude #__TBC

def get_action(q_matrix, current_altitude, current_state, max_q_value):
	results = np.where(q_matrix[current_altitude, current_state] == max_q_value)
	results = results[0]
	if results.size!=1:
		random_seed.shuffle(results)
		action = results[0]
	else: 
		action = results[0]
	return action

