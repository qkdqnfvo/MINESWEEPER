import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Activation, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from skimage.transform import resize
from skimage.color import rgb2gray
import time


def pre_processing(observe):
    rm_img = observe[20:195,:,:]
    processed_observe = np.uint8(resize(rgb2gray(rm_img), (84, 84), mode='constant') * 255)
    return processed_observe


class DQNAgent:
    def __init__(self, action_size):
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        self.learning_rate = 1e-3
        self.epsilon = 0.02
        # self.no_op_steps = 10

        self.model = self.build_model()
        self.model.load_weights("./Invaders_dqn_510.h5")

        self.avg_q_max, self.avg_loss = 0, 0
        self.avg_q_maxs, self.avg_losses = [], []

    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    def build_model(self):
        input_layer = Input(shape=self.state_size)
        hidden_1 = Conv2D(64, (9, 9), strides=(3, 3))(input_layer)
        hidden_1_batch = BatchNormalization()(hidden_1)
        hidden_1_output = Activation('relu')(hidden_1_batch)

        hidden_2 = Conv2D(32, (5, 5), strides=(2, 2))(hidden_1_output)
        hidden_2_batch = BatchNormalization()(hidden_2)
        hidden_2_output = Activation('relu')(hidden_2_batch)

        hidden_3 = Conv2D(32, (3, 3), strides=(1, 1))(hidden_2_output)
        hidden_3_batch = BatchNormalization()(hidden_3)
        hidden_3_output = Activation('relu')(hidden_3_batch)

        flatten_layer = Flatten()(hidden_3_output)

        hidden_4 = Dense(128, activation='relu')(flatten_layer)
        output_layer = Dense(self.action_size)(hidden_4)

        model = Model(input_layer, output_layer)
        model.summary()
        model.compile(loss='mse', optimizer=Adam(self.learning_rate, clipnorm=10.))
        return model


if __name__ == "__main__":
    startTime = time.time()

    env = gym.make('SpaceInvaders-v4')
    agent = DQNAgent(action_size=6)
    scores, episodes = [], []
    
    num_episode = 100
    for e in range(num_episode):
        done = False
        dead = False
        step, score, start_life = 0, 0, 3
        observe = env.reset()
        
        # for _ in range(random.randint(1, agent.no_op_steps)):
        #     observe, _, _, _ = env.step(0)
        
        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2) 
        history = np.reshape([history], (1, 84, 84, 4))
        while not done:
            env.render()
            step += 1

            # 0:'NOOP', 1:'FIRE', 2:'RIGHT', 3:'LEFT', 4:'RIGHTFIRE', 5:'LEFTFIRE'
            action = agent.get_action(history)
            
            if dead:
                action, dead = 1, False

            observe, reward, done, info = env.step(action)

            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            agent.avg_q_max += np.amax(agent.model.predict(np.float32(history / 255.))[0])
            
            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            if reward == 200:
                reward = 0
            reward = reward if not dead or (done and score>=630) else -50

            score += reward

            if dead:
                history = np.stack((next_state, next_state, next_state, next_state), axis=2)
                history = np.reshape([history], (1, 84, 84, 4))
            else:
                history = next_history

            if done:
                print("episode:", e, " score:", score, " average_q: %.5f"%(agent.avg_q_max / float(step)))
                scores.append(score)
                episodes.append(e)
                agent.avg_q_maxs.append(agent.avg_q_max/float(step))
                agent.avg_q_max = 0

        if e % 10 == 0:
            pylab.plot(episodes, agent.avg_q_maxs, 'b')
            pylab.title('q')
            pylab.savefig("./qmax_graph/q.png")
            pylab.clf()
            np.savetxt('./qmax_graph/avg_qmaxs.txt', agent.avg_q_maxs, fmt='%.5f')
            
            pylab.plot(episodes, scores, 'b')
            pylab.title('score')
            pylab.savefig("./score/Invaders_dqn.png")
            pylab.clf()
            np.savetxt('./score/scores.txt', scores, fmt='%d')


    print("elapsed time: ", time.time() - startTime)
