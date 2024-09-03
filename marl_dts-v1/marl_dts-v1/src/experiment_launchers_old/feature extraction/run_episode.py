import gym
import numpy as np
import random



#"PongDeterministic-v4"
#PongNoFrameskip-v4
#Pong-v4
#BoxingNoFrameskip-v4


env = gym.make("Pong-v0", render_mode = 'human') # render_mode = 'human'

episodes = 1

dataset = []

for e in range(episodes):
    env.seed(e)
    obs = env.reset()
    done = False
    count = 0
    
    while not done:
        
        #dataset.append(obs[34:180, 20:-20])
        dataset.append(obs[35:])
        action = random.randint(1,5)
        obs, rew, done, _ = env.step(action)
        

# from PIL import Image
# img = Image.fromarray(dataset[0], 'RGB')
# img.show()