from collections import deque
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Activation, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import Counter
from mss import mss
import sys
# import pylab
import random
import numpy as np
import time
import cv2
import pyautogui
import webbrowser
import pygetwindow as gw
import os
from pynput.keyboard import Key, Controller
import threading
import ctypes

keyboard_button = Controller()

pyautogui.FAILSAFE = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def pre_processing(observe):
    processed_observe = np.uint8(resize(rgb2gray(observe), (130//2, 130//2), mode='constant') * 255)
    return processed_observe

def pre_processing_dead(observe):
    processed_observe = np.uint8(resize(rgb2gray(observe), (24//2, 24//2), mode='constant') * 255)
    return processed_observe

def getImage():
    # bounding_box = {'top': 100, 'left': 12, 'width': 260, 'height': 260} # 중급
    bounding_box = {'top': 100, 'left': 12, 'width': 130, 'height': 130}
    sct = mss()
    sct_img = sct.grab(bounding_box)
    sct_img = np.array(sct_img)
    sct_img = sct_img[:,:,:3]
    return sct_img

def getDead():
    # bounding_box = {'top': 62, 'left': 129, 'width': 24, 'height': 24} # 중급
    bounding_box = {'top': 62, 'left': 65, 'width': 24, 'height': 24}
    sct = mss()
    sct_img = sct.grab(bounding_box)
    sct_img = np.array(sct_img)
    sct_img = sct_img[:,:,:3]
    return sct_img

class DQNAgent:
    def __init__(self, action_size):
        self.state_size = (65, 65, 2)
        self.action_size = action_size
        self.learning_rate = 1e-3
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.02
        self.exploration_steps = 50000
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) / self.exploration_steps
        self.batch_size = 256
        self.train_start = 1000
        # self.train_start = 10000
        self.update_target_rate = 10000
        self.discount_factor = 0.99
        self.memory = deque(maxlen=50000)
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.avg_q_max, self.avg_loss = 0, 0
        self.avg_q_maxs, self.avg_losses = [], []

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    def append_sample(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step
        batch = random.sample(self.memory, self.batch_size)
        history = np.zeros((self.batch_size, 65, 65, 2), dtype=np.float32)
        next_history = np.zeros((self.batch_size, 65, 65, 2), dtype=np.float32)
        actions, rewards, dead = [], [], []

        for i in range(self.batch_size):
            history[i] = batch[i][0][0] / 255.
            actions.append(batch[i][1])
            rewards.append(batch[i][2])
            next_history[i] = batch[i][3][0] / 255.
            dead.append(batch[i][4])

        target = self.model.predict(history)
        target_predicts = self.target_model.predict(next_history)

        max_q = np.amax(target_predicts, axis=1)
        for i in range(self.batch_size):
            if dead[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * max_q[i]
        h = self.model.fit(history, target, batch_size=self.batch_size, epochs=1, verbose=0)
        self.avg_loss += h.history['loss'][0]


    def build_model(self):
        input_layer = Input(shape=self.state_size)
        hidden_1 = Conv2D(32, (9, 9), strides=(3, 3), kernel_initializer='he_uniform')(input_layer)
        hidden_1_output = Activation('relu')(hidden_1)

        hidden_2 = Conv2D(64, (5, 5), strides=(2, 2), kernel_initializer='he_uniform')(hidden_1_output)
        hidden_2_output = Activation('relu')(hidden_2)

        hidden_3 = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer='he_uniform')(hidden_2_output)
        hidden_3_output = Activation('relu')(hidden_3)

        flatten_layer = Flatten()(hidden_3_output)

        hidden_4 = Dense(512, activation='relu', kernel_initializer='he_uniform')(flatten_layer)
        output_layer = Dense(self.action_size, kernel_initializer='he_uniform')(hidden_4)

        model = Model(input_layer, output_layer)
        model.summary()
        model.compile(loss='mse', optimizer=Adam(self.learning_rate, clipnorm=1.))
        return model

# for i in mouse_list:

class Env():
    def __init__(self):
        self.mouse_list = []
        for i in range(8):
            for j in range(8):
                self.mouse_list.append((13+130/8*j+130/16, 101+130/8*i+130/16))
        self.end = cv2.imread('D:/Workspace/My/Python/project/minesweeper/1/img/dead.png', cv2.IMREAD_UNCHANGED)
        self.clear = cv2.imread('D:/Workspace/My/Python/project/minesweeper/1/img/clear.png', cv2.IMREAD_UNCHANGED)
        
    def check_end(self, array):
        a = (array == self.end).flatten()
        if Counter(a)[True] == 144:
            return True
        else:
            return False

    def check_clear(self, array):
        a = (array == self.clear).flatten()
        if Counter(a)[True] == 144:
            return True
        else:
            return False

    def step(self, action):
        pyautogui.click(x=self.mouse_list[action][0], y=self.mouse_list[action][1], button='left')
        time.sleep(0.05)
        observe = getImage()
        observe_gray = pre_processing(observe)
        if self.check_end(pre_processing_dead(getDead())):
            reward = -1
            dead = True
        else:
            reward = 1
            dead = False
        return observe_gray, reward, dead
        
    def reset(self):
        pyautogui.click(x=65+12, y=62+12, button='left')
        time.sleep(0.05)
        pyautogui.click(x=65+12, y=62+12, button='left')
        return getImage()


if __name__ == "__main__":
    # np.set_printoptions(threshold=sys.maxsize)
    startTime = time.time()
    last_time = startTime
    action_size = 64
    agent = DQNAgent(action_size)
    global_step = 0
    scores, episodes = [], []
    env = Env()
    num_episode = 10000
    for e in range(num_episode):
        dead = False
        step, score = 0, 0
        observe = env.reset()   
        state = pre_processing(observe)
        history = np.stack((state, state), axis=2) 
        history = np.reshape([history], (1, 65, 65, 2))
        while not dead:
            # winname = "test"
            # cv2.namedWindow(winname)   # create a named window
            # cv2.moveWindow(winname, 0, 500)   # Move it to (40, 30)
            
            action = agent.get_action(history)
            observe, reward, dead = env.step(action)
            next_state = pre_processing(observe)
            # cv2.imshow(winname, next_state)
            next_state = np.reshape([next_state], (1, 65, 65, 1))
            next_history = np.append(next_state, history[:, :, :, :1], axis=3)
            if Counter((next_history[:,:,:,0] == next_history[:,:,:,1]).flatten())[True] == 4225:
                reward = -1
            score += reward
            agent.append_sample(history, action, reward, next_state, dead)
            global_step += 1
            step += 1
            agent.avg_q_max += np.amax(agent.model.predict(np.float32(history / 255.))[0])

            if len(agent.memory) >= agent.train_start:
                agent.train_model()
                if global_step % agent.update_target_rate == 0:
                    agent.update_target_model()
            if dead:
                history = np.stack((next_state, next_state), axis=2)
                history = np.reshape([history], (1, 65, 65, 2))
            else:
                history = next_history

            if env.check_clear(pre_processing_dead(getDead())):
                print('Game Clear')
                break

            if dead:
                agent.avg_q_max = agent.avg_q_max / float(step)
                agent.avg_loss = agent.avg_loss / float(step)
                print("episode: %d|"%e, "score: %3d|"%int(score), "mem_len: %d|"%len(agent.memory), "epsilon: %.2f|"%agent.epsilon,
                    "global_step: %d|"%global_step, "avg_q: %.2f|"%agent.avg_q_max, "avg_loss: %.2f|"%agent.avg_loss)
                scores.append(score)
                episodes.append(e)
                agent.avg_losses.append(agent.avg_loss)
                agent.avg_q_maxs.append(agent.avg_q_max)
                agent.avg_loss = 0
                agent.avg_q_max = 0

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                cv2.destroyAllWindows()
                break

        if e % 10 == 0 and e != 0:
            current_time = time.time()
            epi_time = current_time - last_time
            total_time = current_time - startTime
            last_time = current_time
            print("10_episode_elapsed_time(min): %.3f"%(epi_time/60))
            print("total_elapsed_time(min): %.3f"%(total_time/60))
            # agent.model.save_weights("C:/Workspace/dino_smol/h5/Dino_dqn_%d.h5"%e)

            # pylab.plot(episodes, agent.avg_losses, 'b')
            # pylab.title('loss')
            # pylab.savefig("C:/Workspace/dino_smol/loss_graph/loss_%d.png"%e)
            # pylab.clf()
            # np.savetxt('C:/Workspace/dino_smol/loss_graph/text/avg_losses_%d.txt'%e, agent.avg_losses, fmt='%.5f')

            # pylab.plot(episodes, agent.avg_q_maxs, 'b')
            # pylab.title('qmax')
            # pylab.savefig("C:/Workspace/dino_smol/qmax_graph/q_%d.png"%e)
            # pylab.clf()
            # np.savetxt('C:/Workspace/dino_smol/qmax_graph/text/avg_qmaxs_%d.txt'%e, agent.avg_q_maxs, fmt='%.5f')
            
            # pylab.plot(episodes, scores, 'b')
            # pylab.title('score')
            # pylab.savefig("C:/Workspace/dino_smol/score/Dino_dqn_%d.png"%e)
            # pylab.clf()
            # np.savetxt('C:/Workspace/dino_smol/score/text/scores_%d.txt'%e, scores, fmt='%d')

    print("elapsed_time(min): %.3f"%((time.time() - startTime) / 60))