import numpy as np
from PIL import Image
import cv2


dataset = np.load('pong_newdata.npy')[30:200:3]

#dataset = np.load('pong_data.npy')[:190]

def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """
    Adds salt and pepper noise to an image.
    
    Parameters:
    - image: Input image.
    - salt_prob: Probability of adding salt (white) noise.
    - pepper_prob: Probability of adding pepper (black) noise.
    
    Returns:
    - Noisy image.
    """
    noisy_image = np.copy(image)
    # Salt noise
    num_salt = np.ceil(salt_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    noisy_image[tuple(coords)] = 255
    
    # Pepper noise
    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    noisy_image[tuple(coords)] = 0
    
    return noisy_image


def add_speckle_noise(image, mean=0, sigma=0.1):
    """
    Adds speckle noise to a numpy array image.
    
    Parameters:
    - image_np: Numpy array representing an input image.
    - mean: Mean of the Gaussian distribution of noise.
    - sigma: Standard deviation of the Gaussian distribution of noise.
    
    Returns:
    - Numpy array representing the noisy image.
    """
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + image * gauss
    noisy_image = np.clip(noisy_image, 0, 255)  # Ensure values are within a valid range
    return noisy_image.astype(np.uint8)

def add_gaussian_noise(image, mean=0, sigma=0.1):
    """
    Adds Gaussian noise to a numpy array image.
    
    Parameters:
    - image_np: Numpy array representing an input image.
    - mean: Mean of the Gaussian distribution.
    - sigma: Standard deviation of the Gaussian distribution.
    
    Returns:
    - Numpy array representing the noisy image.
    """
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, 0, 255)  # Ensure values are within valid range
    return noisy_image.astype(np.uint8)



resized_images = []
noisy_images = []


width = 96
height = 96

for im in dataset:
    
    
    resized_im = cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA)
    #noise = add_salt_and_pepper_noise(resized_im, 0.05, 0.05)
    #noise = add_speckle_noise(resized_im)
    
    img = Image.fromarray(resized_im, 'RGB')
    #img.show()
    
    resized_images.append(resized_im)
    #noisy_images.append(noise)

np.save('small_pong_resized',np.array(resized_images))
#np.save('noisy',np.array(noisy_images))
    

    