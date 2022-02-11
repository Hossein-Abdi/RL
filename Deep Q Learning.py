## Hossein Abdi
## Micro-Robot Deep Q-Learning

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import matplotlib.pyplot as plt
import numpy as np
import math
import time
import AP
import BP

## Set the hyperparameters for training:
EPISODES = 1000 # total number of episodes
EXPLORE_EPI_END = int(0.3*EPISODES) # initial exploration when agent will explore and no training
TEST_EPI_START = int(0.9*EPISODES ) # agent will be tested from this episode
EPS_START = 1.0 # e-greedy threshold start value
EPS_END = 0.05 # e-greedy threshold end value
EPS_DECAY = 1+np.log(EPS_END)/(0.6*EPISODES) # e-greedy threshold decay
GAMMA = 0.7 # Q-learning discount factor
LR = 0.001  # NN optimizer learning rate
MINIBATCH_SIZE = 64  # Q-learning batch size
ITERATIONS = 40      # Number of iterations for training
REP_MEM_SIZE = 10000 # Replay Memory size
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

## Neural Network
STATE_SIZE = 6   # X, Y, Z, Phi, Theta, Sai
ACTION_SIZE = 64 # Action
HiddenLayer1 = 25 
HiddenLayer2 = 25 

## Time
Ttot = 10       # Total Time
dt = 0.1        # Time Step

## Initial Orientation
Phi0 = 0   
Theta0 = 0 
Sai0 = 0   

## Target
X_d = -1.47
Y_d = -1.4
Z_d = -1.58
Accuracy = 0.2

## Value of Parameters
r = 1       # radius of disk (um)
L = 10      # Length of Link (um)
Umax = +5   # maximum Length of link (um)
Umin = -5   # minimum Length of link (um)
Lmax = L + Umax
Lmin = L + Umin
dL = 5      # Velocity of Linear Actuator (um/s)
mu = 1e-9   # Dynamic Viscosity of Fluid 
C0 = 8*math.pi*mu

## Data Save
Data_dis = []
Data_Rew = []
Data_CumRew = []
Data_epsilon = []
Data_Qmax = []

                             
## Action = [T1,T2,T3,u1,u2,u3]
Action = [[0, 0, 0, 0, 0, 0],
          [math.pi/2, 0, 0, 0, 0, 0],
          [0, math.pi/2, 0, 0, 0, 0],
          [0, 0, math.pi/2, 0, 0, 0],
          [math.pi/2, math.pi/2, 0, 0, 0, 0],
          [math.pi/2, 0, math.pi/2, 0, 0, 0],
          [0, math.pi/2, math.pi/2, 0, 0, 0],
          [math.pi/2, math.pi/2, math.pi/2, 0, 0, 0],
          [0, 0, 0, Umax, 0, 0],
          [math.pi/2, 0, 0, Umax, 0, 0],
          [0, math.pi/2, 0, Umax, 0, 0],
          [0, 0, math.pi/2, Umax, 0, 0],
          [math.pi/2, math.pi/2, 0, Umax, 0, 0],
          [math.pi/2, 0, math.pi/2, Umax, 0, 0],
          [0, math.pi/2, math.pi/2, Umax, 0, 0],
          [math.pi/2, math.pi/2, math.pi/2, Umax, 0, 0],
          [0, 0, 0, 0, Umax, 0],
          [math.pi/2, 0, 0, 0, Umax, 0],
          [0, math.pi/2, 0, 0, Umax, 0],
          [0, 0, math.pi/2, 0, Umax, 0],
          [math.pi/2, math.pi/2, 0, 0, Umax, 0],
          [math.pi/2, 0, math.pi/2, 0, Umax, 0],
          [0, math.pi/2, math.pi/2, 0, Umax, 0],
          [math.pi/2, math.pi/2, math.pi/2, 0, Umax, 0],
          [0, 0, 0, 0, 0, Umax],
          [math.pi/2, 0, 0, 0, 0, Umax],
          [0, math.pi/2, 0, 0, 0, Umax],
          [0, 0, math.pi/2, 0, 0, Umax],
          [math.pi/2, math.pi/2, 0, 0, 0, Umax],
          [math.pi/2, 0, math.pi/2, 0, 0, Umax],
          [0, math.pi/2, math.pi/2, 0, 0, Umax],
          [math.pi/2, math.pi/2, math.pi/2, 0, 0, Umax],
          [0, 0, 0, Umax, Umax, 0],
          [math.pi/2, 0, 0, Umax, Umax, 0],
          [0, math.pi/2, 0, Umax, Umax, 0],
          [0, 0, math.pi/2, Umax, Umax, 0],
          [math.pi/2, math.pi/2, 0, Umax, Umax, 0],
          [math.pi/2, 0, math.pi/2, Umax, Umax, 0],
          [0, math.pi/2, math.pi/2, Umax, Umax, 0],
          [math.pi/2, math.pi/2, math.pi/2, Umax, Umax, 0],
          [0, 0, 0, Umax, 0, Umax],
          [math.pi/2, 0, 0, Umax, 0, Umax],
          [0, math.pi/2, 0, Umax, 0, Umax],
          [0, 0, math.pi/2, Umax, 0, Umax],
          [math.pi/2, math.pi/2, 0, Umax, 0, Umax],
          [math.pi/2, 0, math.pi/2, Umax, 0, Umax],
          [0, math.pi/2, math.pi/2, Umax, 0, Umax],
          [math.pi/2, math.pi/2, math.pi/2, Umax, 0, Umax],
          [0, 0, 0, 0, Umax, Umax],
          [math.pi/2, 0, 0, 0, Umax, Umax],
          [0, math.pi/2, 0, 0, Umax, Umax],
          [0, 0, math.pi/2, 0, Umax, Umax],
          [math.pi/2, math.pi/2, 0, 0, Umax, Umax],
          [math.pi/2, 0, math.pi/2, 0, Umax, Umax],
          [0, math.pi/2, math.pi/2, 0, Umax, Umax],
          [math.pi/2, math.pi/2, math.pi/2, 0, Umax, Umax],
          [0, 0, 0, Umax, Umax, Umax],
          [math.pi/2, 0, 0, Umax, Umax, Umax],
          [0, math.pi/2, 0, Umax, Umax, Umax],
          [0, 0, math.pi/2, Umax, Umax, Umax],
          [math.pi/2, math.pi/2, 0, Umax, Umax, Umax],
          [math.pi/2, 0, math.pi/2, Umax, Umax, Umax],
          [0, math.pi/2, math.pi/2, Umax, Umax, Umax],
          [math.pi/2, math.pi/2, math.pi/2, Umax, Umax, Umax]]



## Classes
class QNet(nn.Module): 
    # Input to the network is a 6-dimensional state vector and the output a 64-dimensional vector
    
    def __init__(self, state_space_dim, action_space_dim):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(state_space_dim, HiddenLayer1)
        self.l2 = nn.Linear(HiddenLayer1, HiddenLayer2)
        self.l3 = nn.Linear(HiddenLayer2, action_space_dim)
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
    
class ReplayMemory:
    # storing a large number of [State, Action, Next State, Reward] in memory
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    def push(self, s, a , r, ns):
        self.memory.append((FloatTensor([s]),
                    a, # action is already a tensor
                    FloatTensor([ns]),
                    FloatTensor([r])))
        if len(self.memory) > self.capacity:
            del self.memory[0]
    def sample(self, MINIBATCH_SIZE):
        return random.sample(self.memory, MINIBATCH_SIZE)
    def __len__(self):
        return len(self.memory)

class QNetAgent:
    def __init__(self, stateDim, actionDim):
        self.sDim = stateDim
        self.aDim = actionDim
        self.model = QNet(self.sDim, self.aDim) # Instantiate the NN model, loss and optimizer for training the agent
        if use_cuda:
            self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), LR)
        self.lossCriterion = torch.nn.MSELoss()
        self.memory = ReplayMemory(REP_MEM_SIZE) # Instantiate the Replay Memory for storing agent's experiences
        # Initialize internal variables
        self.steps_done = 0
        
    def select_action(self, state):
        # Select action based on epsilon-greedy policy
        p = random.random() # generate a random number between 0 and 1
        self.steps_done += 1
        if p > self.epsilon:
            # if the agent is in 'exploitation mode' select optimal action
            # based on the highest Q value returned by the trained NN
            with torch.no_grad():
                return self.model(FloatTensor(state)).data.max(1)[1].view(1, 1)
        else:
            # if the agent is in the 'exploration mode' select a random action
            return LongTensor([[random.randrange(action_size)]])
        
    def run_episode(self, e):
        print("New Episode = ",e)
        # reset the environment at the beginning
        X = [0]; Y = [0]; Z = [0]
        Phi = [Phi0]; Theta = [Theta0]; Sai = [Sai0]
        dX = []; dY = []; dZ = []
        dPhi = []; dTheta = []; dSai = []
        u1 = [0]; u2 = [0]; u3 = [0]
        T1 = []; T2 = []; T3 = []
        du1 = []; du2 = []; du3 = []
        state = [X[0], Y[0], Z[0], Phi[0], Theta[0], Sai[0]]
        done = False
        steps = 0
        # Set the epsilon value for the episode
        if e < EXPLORE_EPI_END:
            self.epsilon = EPS_START
            self.mode = "Exploring"
        elif EXPLORE_EPI_END <= e <= TEST_EPI_START:
            self.epsilon = self.epsilon*EPS_DECAY
            self.mode = "Training"
        elif e > TEST_EPI_START:
            self.epsilon = 0.0
            self.mode = "Testing"
   
        ## Dynamic Simulation
        N = Ttot/dt
        for i in range (int(N)):
            print (i)
            t = i*dt 
            if abs(t%1)<1e-5: 
                state = [X[i], Y[i], Z[i], Phi[i], Theta[i], Sai[i]]
                print("state = [",state[0],",",state[1],",",state[2],"] , [",state[3],",",state[4],",",state[5],"]")
                action = self.select_action(FloatTensor([state])) # Select action based on epsilon-greedy approach
                print("action = ",action.data.cpu().numpy()[0,0])           
            a = action.data.cpu().numpy()[0,0] 
            # Rotary Actuators
            T1 = T1 + [Action[a][0]]
            T2 = T2 + [Action[a][1]]
            T3 = T3 + [Action[a][2]]
            # Linear Actuators
            if Action[a][3]>u1[i]:
                du1 = du1 + [dL]
            if Action[a][3]==u1[i]:
                du1 = du1 + [0]
            if Action[a][3]<u1[i]:
                du1 = du1 + [-dL]
            if Action[a][4]>u2[i]:
                du2 = du2 + [dL]
            if Action[a][4]==u2[i]:
                du2 = du2 + [0]
            if Action[a][4]<u2[i]:
                du2 = du2 + [-dL]
            if Action[a][5]>u3[i]:
                du3 = du3 + [dL]
            if Action[a][5]==u3[i]:
                du3 = du3 + [0]
            if Action[a][5]<u3[i]:
                du3 = du3 + [-dL]

            ## Get next state and reward from environment based on current action
            Ap = AP.AP(u1[i],u2[i],u3[i],T1[i],T2[i],T3[i],X[i],Y[i],Z[i],Phi[i],Theta[i],Sai[i])
            Bp = BP.BP(u1[i],u2[i],u3[i],T1[i],T2[i],T3[i],du1[i],du2[i],du3[i],X[i],Y[i],Z[i],Phi[i],Theta[i],Sai[i])
            solP = np.dot(np.linalg.inv(Ap),np.array(Bp))
            # Disturbance
            D_X = (0.5+0.75*(math.sin(20*math.pi*t)+math.cos(25*math.pi*t)))*1e-3
            D_Y = (0.8+0.65*(math.cos(15*math.pi*t)+math.sin(30*math.pi*t)))*1e-3
            D_Z = (0.6+0.85*(math.sin(20*math.pi*t)+math.cos(25*math.pi*t)))*1e-3
            D_Phi = (0.05+1.5*(math.cos(30*math.pi*t)+math.sin(35*math.pi*t)))*1e-3
            D_Theta = (0.04+1*(math.sin(40*math.pi*t)+math.cos(25*math.pi*t)))*1e-3
            D_Sai = (0.06+0.9*(math.cos(20*math.pi*t)+math.sin(15*math.pi*t)))*1e-3
            # Linear and Angular Velocity
            dX = dX + list(solP[0]+D_X)
            dY = dY + list(solP[1]+D_Y)
            dZ = dZ + list(solP[2]+D_Z)
            dPhi = dPhi + list(solP[3]+D_Phi)
            dTheta = dTheta + list(solP[4]+D_Theta)
            dSai = dSai + list(solP[5]+D_Sai)
            # Next State
            X = X + [X[i] + dX[i]*dt]
            Y = Y + [Y[i] + dY[i]*dt]
            Z = Z + [Z[i] + dZ[i]*dt]
            Phi = Phi + [Phi[i] + dPhi[i]*dt]
            Theta = Theta + [Theta[i] + dTheta[i]*dt]
            Sai = Sai + [Sai[i] + dSai[i]*dt]
            u1 = u1 + [u1[i] + du1[i]*dt]
            u2 = u2 + [u2[i] + du2[i]*dt]
            u3 = u3 + [u3[i] + du3[i]*dt]
            # Reward
            if abs((t+dt)%1)<1e-5 and t>0:
                Distance = math.sqrt((X[i+1]-X_d)**2+(Y[i+1]-Y_d)**2+(Z[i+1]-Z_d)**2)
                print("Distance = ",Distance)
                if abs(Distance) <= Accuracy:
                    Reward = +100-100*Distance
                    next_state, reward, done = [[X[i+1],Y[i+1],Z[i+1],Phi[i+1],Theta[i+1],Sai[i+1]], Reward, True] 
                else:
                    Reward = 20*math.exp(-0.5*Distance)
                    next_state, reward, done = [[X[i+1],Y[i+1],Z[i+1],Phi[i+1],Theta[i+1],Sai[i+1]], Reward, False]
                print("Reward = ",Reward)
                print("Next State = [",next_state[0],",",next_state[1],",",next_state[2],"] , [",next_state[3],",",next_state[4],",",next_state[5],"]")
                
                ## push experience into replay memory
                self.memory.push(state, action, reward, next_state)
                state = next_state
                steps += 1             
           
            # if initial exploration is finished train the agent
            if EXPLORE_EPI_END <= e <= TEST_EPI_START:
                self.learn()
               
            if done: # Print information after every episode
                print("Mission Accomplished!")
                break

        global Data_dis
        global Data_Rew
        global Data_CumRew 
        global Data_epsilon
         
        # save data
        Data_dis = Data_dis + [Distance]
        Data_Rew = Data_Rew + [Reward]
        Data_CumRew = Data_CumRew +[sum(Data_Rew)]
        Data_epsilon = Data_epsilon + [self.epsilon]
           
    def learn(self):
        # Training the DNN using the randomly sampled of [State, Action, Next State, Reward] from the ReplayStore memory.        
        if len(self.memory) < MINIBATCH_SIZE:
            return
        for i in range(ITERATIONS):
            # minibatch is generated by random sampling from experience replay memory
            experiences = self.memory.sample(MINIBATCH_SIZE)
            batch_state, batch_action, batch_next_state, batch_reward = zip(*experiences)
            # extract experience information for the entire minibatch
            batch_state = torch.cat(batch_state)
            batch_action = torch.cat(batch_action)
            batch_reward = torch.cat(batch_reward)
            batch_next_state = torch.cat(batch_next_state)
            # current Q values are estimated by NN for all actions
            current_q_values = self.model(batch_state).gather(1, batch_action)
            # expected Q values are estimated from actions which gives maximum Q value
            max_next_q_values = self.model(batch_next_state).detach().max(1)[0]
            expected_q_values = batch_reward + (GAMMA * max_next_q_values)
            # loss is measured from error between current and newly expected Q values
            loss = self.lossCriterion(current_q_values, expected_q_values.unsqueeze(1))
            # backpropagation of loss for NN training
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

                        
if __name__ == "__main__":
    StartTime = time.time()
    state_size = STATE_SIZE
    action_size = ACTION_SIZE
    # Instantiate the RL Agent
    agent = QNetAgent(state_size, action_size)
    for e in range(EPISODES): # Train the agent
        agent.run_episode(e)
    print('Complete')
    EndTime = time.time()
    print("Total Time = ",EndTime-StartTime," s")

    
