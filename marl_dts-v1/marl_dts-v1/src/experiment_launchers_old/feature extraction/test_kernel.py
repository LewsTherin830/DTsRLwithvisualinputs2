import numpy as np
import cv2
from PIL import Image





dataset = np.load("resized.npy")
im_w = 96



def find_indexes_2d(array, value):
    indexes = []
    for i, row in enumerate(array):
        for j, element in enumerate(row):
            if element == value:
                indexes.append((j, i))
    if(len(indexes) > 0):
        indexes = np.array(indexes)
        return np.floor(np.mean(indexes, axis = 0))
    else:
        return np.array([0,0])
    

# #preprocessing observation
# #f1 = find_indexes_2d(observation, 213) #other player

def find_coord_color(d):
    ball = find_indexes_2d(d[:-10,:,0], 236) #the ball
    paddle = find_indexes_2d(d[:,:,0], 92)  #our player
    
    return [*ball,*paddle]
    
    
def find_coord_conv(d):
    
    patches = [None, None]
    ball = -1 * np.ones((5, 5, 5))
    ball[1:4, 1:4] = 1
    patches[0] = ball
    rack = -1 * np.ones((5, 5, 5))
    rack[2:, 2:4, 1] = 1
    patches[1] = rack
    
    features = []
    for weight in patches:
        obs = d / 255
        out = np.zeros(obs.shape[:2])
        for ch in range(3):
            out += cv2.filter2D(obs[:, :, ch], -1, weight[:, :, ch])
        tempf = np.argmax(out.flatten())
        x = tempf % im_w
        y = tempf // im_w
        features.extend([x, y])
    features = np.array(features)
    
    return features


def find_coord_kernel(d):
    
    patches = [None, None]
    ball = -1 * np.ones((5, 5, 5))
    patches[0] = ball
    rack = -1 * np.ones((5, 5, 5))
    patches[1] = rack
    
    features = []
    for weight in patches:
        obs = d / 255
        out = np.zeros(obs.shape[:2])
        for ch in range(3):
            out += cv2.filter2D(obs[:, :, ch], -1, weight[:, :, ch])
        tempf = np.argmax(out.flatten())
        x = tempf % im_w
        y = tempf // im_w
        features.extend([x, y])
    features = np.array(features)
    
    return features


for d in dataset:
    features = find_coord_color(d) 
    
    d[int(features[1]),int(features[0])] = [255,0,0]
    d[int(features[3]),int(features[2])] = [0,0,255]
    img = Image.fromarray(d, 'RGB')
    img.show()
    
    
    
