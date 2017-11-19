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


def playGame(train_indicator=1):  # 1 means Train, 0 means simply Run
    GAMMA = 0.99
    max_steps = 10000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0

    nEpisodes = 100

    # Generate a Torcs environment
    env = TorcsEnv(vision=False, throttle=False, gear_change=False)

    tf.reset_default_graph()
    # These lines establish the feed-forward part of the network used to choose actions
    nInputs = 29
    nOutput = 3

    inputs1 = tf.placeholder(shape=[1, nInputs], dtype=tf.float32)
    W = tf.Variable(tf.random_uniform([nInputs, nOutput], 0, 0.01))
    Qout = tf.matmul(inputs1, W)
    predict = tf.argmax(Qout, 1)

    # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
    nextQ = tf.placeholder(shape=[1, nOutput], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(nextQ - Qout))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    updateModel = trainer.minimize(loss)

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        print("TORCS Experiment Start.")
        for i in range(nEpisodes):

            print("Episode : " + str(i))

            if np.mod(i, 3) == 0:
                # relaunch TORCS every 3 episode because of the memory leak error
                ob = env.reset(relaunch=True)
            else:
                ob = env.reset()

            s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,
                             ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))

            total_reward = 0.
            for j in range(max_steps):

                action = 1

                # Choose an action by greedily (with e chance of random action) from the Q-network
                action, allQ = sess.run([predict, Qout], feed_dict={inputs1: s_t.reshape(1, 29)})
                # if np.random.rand(1) < epsilon:
                # action[0] = env.action_space.sample()

                ob, r_t, done, info = env.step(action[0])

                s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX,
                                  ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))

                # Obtain the Q' values by feeding the new state through our network
                Q1 = sess.run(Qout, feed_dict={inputs1: s_t1.reshape(1, 29)})
                # Obtain maxQ' and set our target value for chosen action.
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0, action[0]] = r_t + GAMMA * maxQ1

                if (train_indicator):
                    # Train our network using target and predicted Q values
                    _, W1 = sess.run([updateModel, W], feed_dict={
                        inputs1: s_t.reshape(1, 29), nextQ: targetQ})

                total_reward += r_t
                s_t = s_t1

                print("Episode", i, "Step", step, "Action", action[0], "Reward", r_t)

                step += 1

                if done:
                    epsilon = 1. / ((i / 50) + 10)
                    break

            print("TOTAL REWARD @ " + str(i) + "-th Episode  : Reward " + str(total_reward))
            print("Total Step: " + str(step))
            print("")

        env.end()  # This is for shutting down TORCS
        print("Finish.")


if __name__ == "__main__":
    playGame()
