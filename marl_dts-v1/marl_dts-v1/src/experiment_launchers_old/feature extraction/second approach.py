from time import time
import random
import numpy as np
from PIL import Image
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from matplotlib import pyplot
import numpy as np
from skimage import io, color
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#global_var
#initializing the dataset
dataset1 = np.load('resized.npy')
dataset = np.load('small_pong_resized.npy')
#noise = np.load('noisy.npy')


#query variables
query_shape = (5,5)
stride = 1


def partition_image(image):
    
    h, w, _ = image.shape
    width, height = query_shape
    img_patches = []
    
    for j in range(0, h - height, height):
        for i in range(0, w - width, width):
            img_patches.append(image[j:j+height, i:i+width].flatten())
    img_patches = np.array(img_patches)
    
    return img_patches
    
    

def show_im(data):
    
    img = Image.fromarray(data, 'RGB')
    img.show()
    
    
def show_query(q1):
    
    height, width = query_shape
    img = Image.fromarray(q1.reshape(height,width,3), 'RGB')
    img.show()


def cosine_similarity(a, b):
    
    a = np.array(a, dtype = 'int32')
    b = np.array(b, dtype = 'int32')
    
    return np.dot(a, b)/(np.linalg.norm(a, axis=1) * np.linalg.norm(b))
    
    
def attention(observation, query):
    """
    Uses content-wise (i.e., cosine-similarity) attention to compute features

    :observation: The observation given from the environment (an image)
    :queries: A list of queries
    :config: The configuration
    :returns: A list of (x, y) features
    """
    
    h, w, _ = observation.shape
    height, width = query_shape
    img_patches = []
    
    # q = queries.reshape(3,5,5,3)[0]
    # print(q[:,:,0])
    # from matplotlib import pyplot as plt
    # plt.imshow(res, interpolation='nearest')
    # plt.show()
    
    for j in range(0, h - height, stride):
        for i in range(0, w - width, stride):
            img_patches.append(observation[j:j+height, i:i+width].flatten())
    img_patches = np.array(img_patches)
    

    w_locations = np.floor((w - width) / stride)
    
    attention_scores = cosine_similarity(img_patches, query)
    best = np.argmax(attention_scores)
    #print(attention_scores)

    x = best % w_locations
    y = best // w_locations

    return [x,y]


def evaluate(queries):
    
    features = []
    
    for obs in dataset:
        f = attention(obs, queries)
        features.append(f)
        
    features = np.array(features)
    #print(features)
    
    #fitness =  max(np.std(features[:,0]),np.std(features[:,1])) #fitness based on standard deviation
    
    diff = np.sum((features[:-1]-features[1:])**2)
    
    if diff == 0:
        fitness = 0
    else:
        fitness = 1/diff
    
    print(fitness)
    
    return fitness, features.flatten()




if __name__ == "__main__":
    
    
    image = dataset1[5]
    img = Image.fromarray(image, 'RGB')
    img.show()
    
    img_patches = partition_image(image)
    height, width = query_shape
    query_len = height*width*3
    
    fitnesses = []
    features = []
    
    for q in img_patches:
        query = np.array(q)
        fitness, feature = evaluate(query)
        fitnesses.append(fitness)
        features.append(feature)
        
    features = np.array(features)
    ind_max = np.argmax(fitnesses)
    max_ = fitnesses[ind_max]
    
    show_query(img_patches[ind_max])
    
    zipped_pairs = zip(img_patches, fitnesses)
    sorted_list = sorted(zipped_pairs, key=lambda x: x[1])
    z = [x for x, y in sorted_list]
    
    
    # model = KMeans(n_clusters=4)
    
    # print(features.shape)
    # # fit the model
    # model.fit(features)
    # # assign a cluster to each example
    # yhat = model.predict(features)
    
    # zipped_pairs = zip(yhat, fitnesses)
    # sorted_list = sorted(zipped_pairs, key=lambda x: x[1])
    # f = [x for x, y in sorted_list]
    
    
    # def extract_color_features(image_patches):
    #     features = []
    #     for patch in image_patches:
    #         # Convert patch to LAB color space
    #         patch = patch.reshape(height,width,3)
    #         lab_patch = color.rgb2lab(patch)
    #         # Compute histogram for each channel
    #         hist_l, _ = np.histogram(lab_patch[:,:,0], bins=256, range=(0, 256))
    #         hist_a, _ = np.histogram(lab_patch[:,:,1], bins=256, range=(-128, 128))
    #         hist_b, _ = np.histogram(lab_patch[:,:,2], bins=256, range=(-128, 128))
    #         # Concatenate histograms to form feature vector
    #         hist_features = np.concatenate((hist_l, hist_a, hist_b))
    #         features.append(hist_features)
    #     return np.array(features)
    
    # ff = extract_color_features(img_patches)
    
    # model = KMeans(n_clusters=3)
    
    # # fit the model
    # model.fit(ff)
    # # assign a cluster to each example
    # yhat = model.predict(ff)
    
    # zipped_pairs = zip(yhat, fitnesses)
    # sorted_list = sorted(zipped_pairs, key=lambda x: x[1])
    # f = [x for x, y in sorted_list]
    
    

    
    
    


