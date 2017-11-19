from gym_torcs_dqn import TorcsEnv
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
from keras.engine.training import collect_trainable_weights
import json


import timeit


def playGame(train_indicator=1):    #1 means Train, 0 means simply Run
    
    max_steps = 10000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0

    nEpisodes = 100
    
    # Generate a Torcs environment
    env = TorcsEnv(vision=False, throttle=False,gear_change=False)

    print("TORCS Experiment Start.")
    for i in range(nEpisodes):

        print("Episode : " + str(i))

        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
     
        total_reward = 0.
        for j in range(max_steps):
            
            action = 1
            
            

            ob, r_t, done, info = env.step(action)

            s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        
            total_reward += r_t
            s_t = s_t1
        
            print("Episode", i, "Step", step, "Action", action, "Reward", r_t)
        
            step += 1

            if done:
                break


        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    playGame()
