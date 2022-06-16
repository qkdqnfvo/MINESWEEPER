from collections import deque
from tabnanny import check
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import Counter
from mss import mss
import sys
import pylab
import random
import numpy as np
import time
import cv2
import pyautogui
import webbrowser
import pygetwindow as gw
import os
import warnings
pyautogui.FAILSAFE = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.simplefilter('ignore', UserWarning)
sys.coinit_flags = 2

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

def check_end(array):
    end = cv2.imread('C:/workspace/mine/beg/dead.png', cv2.IMREAD_UNCHANGED)
    a = (array == end).flatten()
    if Counter(a)[True] == 144:
        return True
    else:
        return False


def check_clear(array):
    end = cv2.imread('C:/workspace/mine/beg/clear.png', cv2.IMREAD_UNCHANGED)
    a = (array == end).flatten()
    if Counter(a)[True] == 144:
        return True
    else:
        return False

# mouse_list = []
# for i in range(16):
#     for j in range(16):
#         mouse_list.append((12+260/16*j+260/32, 100+260/16*i+260/32))
# # mouse_list = np.array(mouse_list).resize(16, 16)
# for i in mouse_list:
#     pyautogui.click(x=i[0], y=i[1], button='right')


# mouse_list = []
# for i in range(8):
#     for j in range(8):
#         mouse_list.append((13+130/8*j+130/16, 101+130/8*i+130/16))
# # mouse_list = np.array(mouse_list).resize(16, 16)
# for i in mouse_list:
#     pyautogui.click(x=i[0], y=i[1], button='right')


while True:
    winname1 = "test"
    cv2.namedWindow(winname1)   # create a named window
    cv2.moveWindow(winname1, 260, 0)   # Move it to (40, 30)

    
    winname2 = "dead"
    cv2.namedWindow(winname2)   # create a named window
    cv2.moveWindow(winname2, 520, 0)   # Move it to (40, 30)

    # observe = pre_processing(observe)
    # cv2.imshow(winname, observe)
    a = getImage()
    a = pre_processing(a)
    b = getDead()
    b = pre_processing_dead(b)
    # a = cv2.imread('C:/workspace/test/test1.png', cv2.IMREAD_UNCHANGED)
    
    # np.savetxt('C:/Workspace/test/end.txt', a, fmt='%d')
    # print(check_end(a))
    cv2.imshow(winname1, a)
    cv2.imshow(winname2, b)
    print('end: ', check_end(b), 'clear: ', check_clear(b))
    # cv2.imwrite('C:/workspace/mine/clear.png', b)

    

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break