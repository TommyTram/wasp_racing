# OpenGym CartPole-v0
# -------------------
#
# This code demonstrates use a full DQN implementation
# to solve OpenGym CartPole-v0 problem.
#
# Made as part of blog series Let's make a DQN, available at: 
# https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
# 
# author: Jaromir Janisch, 2016

import random, numpy, math, gym, sys
from keras import backend as K

import tensorflow as tf

from gym_torcs_dqn import TorcsEnv

#----------
HUBER_LOSS_DELTA = 1.0
LEARNING_RATE = 0.00025

#----------
def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)   # Keras does not cover where function in tensorflow :-(

    return K.mean(loss)

#-------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        self.model_ = self._createModel() 

    def _createModel(self):
        model = Sequential()

        #model.add(Dense(units=64, activation='relu', input_dim=stateCnt))
        #model.add(Dense(units=actionCnt, activation='linear'))
        model.add(Dense(512, input_dim=stateCnt))
        model.add(Activation('relu'))
        model.add(Dense(64, input_dim=stateCnt))
        model.add(Activation('relu'))
        model.add(Dense(actionCnt))
        model.add(Activation('linear'))

        opt = RMSprop(lr=LEARNING_RATE)
        model.compile(loss=huber_loss, optimizer=opt)

        return model

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=64, epochs=epochs, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, self.stateCnt), target=target).flatten()

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)        

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def isFull(self):
        return len(self.samples) >= self.capacity

#-------------------- AGENT ---------------------------
MEMORY_CAPACITY = 10
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay

UPDATE_TARGET_FREQUENCY = 1000

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)
        
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)        

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # debug the Q function in poin S
        if self.steps % 100 == 0:
            S = numpy.array([-0.76404908, -0.18806157, -0.07364678, -0.98308269,  0.46421371,
                             -0.12786118,  0.35635856, -0.58340303,  0.13679167, -0.22210082,
                             0.40436726, -0.53746676, -0.55901542,  0.48374101, -0.34700899,
                             0.1506552 , -0.07217786,  0.8702839 ,  0.40087611,  0.86132241,
                             -0.53149064,  0.93351024, -0.09431621,  0.94650319,  0.43374405,
                             0.79313524, -0.66503553, -0.17781279, -0.70930231])

            pred = agent.brain.predictOne(S)
            print(pred[0])
            sys.stdout.flush()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):    
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([ o[0] for o in batch ])
        states_ = numpy.array([ (no_state if o[3] is None else o[3]) for o in batch ])

        p = agent.brain.predict(states)
        p_ = agent.brain.predict(states_, target=True)

        x = numpy.zeros((batchLen, self.stateCnt))
        y = numpy.zeros((batchLen, self.actionCnt))
        
        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * numpy.amax(p_[i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)


class RandomAgent:
    memory = Memory(MEMORY_CAPACITY)

    def __init__(self, actionCnt):
        self.actionCnt = actionCnt

    def act(self, s):
        return random.randint(0, self.actionCnt-1)

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

    def replay(self):
        pass

#-------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, problem):
        ##self.problem = problem
        ##self.env = gym.make(problem)
        self.env = TorcsEnv(vision=False, throttle=False, gear_change=False)
        self.episodes = 0
        self.steps = 0

    def run(self, agent):
        self.episodes += 1
        if numpy.mod(self.episodes,3) == 0:
            ob = self.env.reset(relaunch=True)
        else:
            ob = self.env.reset()
        s = numpy.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX,
                                  ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
        R = 0 

        while True:            
            # self.env.render()
            self.steps += 1

            a = agent.act(s)

            ob_, r, term, info = self.env.step(a)
            
            s_ = numpy.hstack((ob_.angle, ob_.track, ob_.trackPos, ob_.speedX,
                                  ob_.speedY, ob_.speedZ, ob_.wheelSpinVel / 100.0, ob_.rpm))
            s_ = s_.ravel()
            if term == 1:
                done = True
            else:
                done = False

            if done: # terminal state
                s_ = None

            agent.observe( (s, a, r, s_) )
            agent.replay()            

            s = s_
            R += r

            if done:
                break

        # print("Total reward:", R)
            print("Episode" , self.episodes , "Step", self.steps, "Action", a, "Reward", r)

#-------------------- MAIN ----------------------------
PROBLEM = 'CartPole-v0'
env = Environment(PROBLEM)

stateCnt  = 29#env.env.observation_space.shape[0]
actionCnt = 3#env.env.action_space.n

agent = Agent(stateCnt, actionCnt)
randomAgent = RandomAgent(actionCnt)

try:
    while randomAgent.memory.isFull() == False:
        env.run(randomAgent)
    
    agent.memory.samples = randomAgent.memory.samples
    randomAgent = None
    
    while True:
        env.run(agent)
finally:
    agent.brain.model.save("cartpole-dqn.h5")