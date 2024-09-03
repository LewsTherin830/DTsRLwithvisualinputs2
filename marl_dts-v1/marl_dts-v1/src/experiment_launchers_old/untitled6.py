#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    experiment_launchers.pz_magent_launcher
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This module allows to launch experiments for the MAgent environments
    in PettingZoo.
    This module only manages the case in which we want to evolve only a team
    against random opponents.
    Not suitable for non-teamed environments such as gather, since the fitness
    function cannot give a measure of goodness in such contexts.

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
    
    
    runfile('C:/Users/Kirito/Desktop/marl_dts-master-src/marl_dts-master-src/src/experiment_launchers/pong_gp_no_queries.py', args="C:/Users/Kirito/Desktop/marl_dts-master-src/marl_dts-master-src/src/experiment_launchers/configs/pong_attention_gp.json 42")
"""
import gym
import numpy as np
from Qlearning import Qlearning
from FrameStackingWrapper import FrameStackingWrapper

def find_indexes_2d(array, value):
    indexes = []
    for i, row in enumerate(array):
        for j, element in enumerate(row):
            if element == value:
                indexes.append((j, i))
    if(len(indexes) > 0):
        indexes = np.array(indexes)
        return np.mean(indexes, axis = 0)
    else:
        return np.array([0,0])
    
# def attention(observation):

#     #preprocessing observation
#     observation = observation[34:194,:,0] #cropping unnecessary parts  
    
#     #print(observation.shape)
    
    
#     #f1 = find_indexes_2d(observation, 213) #other player
#     ball_pos = find_indexes_2d(observation, 236) #the ball
#     paddle_pos = find_indexes_2d(observation, 92)  #our player
    
        
#     return np.array([ball_pos,paddle_pos]).flatten()

def attention(observation):
    
    #preprocessing observation
    obs1 = observation[0,34:194,:,0] #cropping unnecessary parts    
    ball_pos1 = find_indexes_2d(obs1, 236) #the ball
    #paddle_pos1 = find_indexes_2d(obs1, 92)  #our player
    
    obs2 = observation[1,34:194,:,0] #cropping unnecessary parts    
    ball_pos2 = find_indexes_2d(obs2, 236) #the ball
    paddle_pos2 = find_indexes_2d(obs2, 92)  #our player
    
    vel = (ball_pos2-ball_pos1)
    if(abs(vel[0]) > 10):
        vel = [0,0]
        
    speed = (vel[0]**2+vel[1]**2)**0.5
    direction = [0,0]
    
    if(speed!= 0):
        direction = vel/speed
    
    #print(direction)

    return np.array([paddle_pos2[1],ball_pos2[0]/160,ball_pos2[1]/160,direction[0],direction[1], 0.5])





def save_lines_to_file(lines, file_path):
    with open(file_path, 'a') as file:  # Open file in append mode
        for line in lines:
            file.write(line + '\n')


def tree_out(input_):
    
    # 0 NOOP
    # 1 FIRE
    # 2 RIGHT
    # 3 LEFT
    
    #yr,xb,yb,vx,vy,_ = input_
    
    weight = np.zeros((6,1))
    weight = np.array([ 0.76042282, -0.57736028,  0.38700166,  0.13287305, -0.59170251, -0.46281745])
    #weight = np.random.uniform(low=-1, high=1, size=6)
    eq1 = input_@weight.T
    
    print(eq1)
    
    if(eq1 > 0):
        return 0
    else:
        return 1
    
    # if(yr > 98.82): #98.25
    
    #     if(yb<159.45):
    #         return 0
    #     else:
    #         #print("hi")
    #         return 1  
    # else:
    #     #return 3
    #     if(yb<140):
    #         return 2
    #     else: 
    #         #print("hello")
    #         return 3
    
    
    
    
    


if __name__ == "__main__":
    
    #"PongDeterministic-v4"
    #PongNoFrameskip-v4
    #Pong-v4
    
    
    env = FrameStackingWrapper(gym.make("PongNoFrameskip-v4", render_mode = 'human'), stack_size=2)
    
    #env = gym.make("PongNoFrameskip-v4", render_mode = 'human')
    
    actions = [3,3,3,3,3,3,3,2,0,0,0,0,0,0,0,0,0,3,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]  
    episodes = 1
    
    for e in range(episodes):
        env.seed(e)
        obs = env.reset()
        done = False
        count = 0
        
        for a in actions:
            obs = attention(obs)
            print(obs[0])
            obs, rew, done, _ = env.step(a)
            
            # if(rew != 0):
            #     count = count + 1
            #     if( count == 3):
            #         done = True
            #print(count)
            #print(mapper.get_Q())
    
