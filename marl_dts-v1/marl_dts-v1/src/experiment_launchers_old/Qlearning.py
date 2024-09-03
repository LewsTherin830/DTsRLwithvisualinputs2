import numpy as np
import random

class Qlearning:

    def __init__(self, states, actions, lr = 0.1, gamma = 0.9, epsilon = 0.05, epsilon_decay = 0.99):
        
        self.states = states
        self.actions = actions
        
        self.no_states = len(states)
        self.no_actions = len(actions)
        
        self.Q = np.zeros((self.no_states,self.no_actions), dtype=float)
        
        # self.Q = np.array([[-0.09,  -0.081,  0.],
        #  [-0.090915,   -0.11378347, -0.11694394],
        #  [-0.04587968, -0.05821669, -0.01882649]])
        
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
        #last values
        self.last_state = states[0]
        self.last_action = actions[0]
        self.last_reward = 0
    
    def choose_action(self, state):
        
        self.update(state)
        
        if(self.epsilon > random.random()):
            a = random.randint(0,self.no_actions - 1)
        else:
            s = self.states.index(state)
            a = np.argmax(self.Q[s])
            
        self.epsilon = self.epsilon*self.epsilon_decay
            
        self.last_state = state
        self.last_action = self.actions[a]
        
        return self.actions[a]
    
    def update(self, state):
        
        s = self.states.index(self.last_state)
        a = self.actions.index(self.last_action)
        r = self.last_reward
        s_next = self.states.index(state)
        
        self.Q[s,a] = self.Q[s,a] + self.lr*(r + self.gamma*np.max(self.Q[s_next])- self.Q[s,a])
    
    def set_reward(self, reward):
        self.last_reward = reward
    
    def get_Q(self):
        return self.Q
        
        
        
        