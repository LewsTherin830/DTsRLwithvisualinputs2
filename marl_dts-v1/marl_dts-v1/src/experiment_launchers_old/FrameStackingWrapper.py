import gym
import numpy as np
from collections import deque

class FrameStackingWrapper(gym.Wrapper):
    def __init__(self, env, stack_size=4, crop_top=35):
        super(FrameStackingWrapper, self).__init__(env)
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)
        self.crop_top = crop_top  # Number of pixels to crop from the top

    def reset(self):
        observation = self.env.reset()
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(observation)
        return np.array(self.frames)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return np.array(self.frames), reward, done, info
    
    def render(self, mode='rgb_array', **kwargs):
        # Fetch the original render
        image = super().render(mode=mode, **kwargs)
        if mode == 'human':
            # Crop the top pixels from the image if it's an rgb_array mode
            cropped_image = image[self.crop_top:, :, :]
            return cropped_image
        else:
            # For other modes, return the image as is
            return image

# Example usage:
# Replace 'YourEnvironmentName' with a valid Gym environment ID, e.g., 'CartPole-v1'
# env = FrameStackingWrapper(gym.make('YourEnvironmentName'), stack_size=4, crop_top=35)
