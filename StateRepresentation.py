# !/usr/bin/env python


from constants import *
import numpy as np
import pickle
from BehaviorPolicy import *
# We want the hashing for tiles to be deterministic. So set random seed.
import random
random.seed(9000)
from tiles import *

# image tiles
NUMBER_OF_PIXEL_SAMPLES = 100
CHANNELS = 4
NUM_IMAGE_TILINGS = 4
NUM_IMAGE_INTERVALS = 4
SCALE_RGB = NUM_IMAGE_INTERVALS / 256.0

IMAGE_START_INDEX = 0

# constants relating to image size recieved
IMAGE_HEIGHT = HEIGHT  # rows
IMAGE_WIDTH = WIDTH  # columns

NUMBER_OF_COLOR_CHANNELS = 3  # red, blue, green
PIXEL_FEATURE_LENGTH = np.power(NUM_IMAGE_INTERVALS, NUMBER_OF_COLOR_CHANNELS) * NUM_IMAGE_TILINGS
PREDICTION_FEATURE_LENGTH = 16
DID_TOUCH_FEATURE_LENGTH = 1
NUMBER_OF_GVFS = 10
NUMBER_OF_ACTIONS = 4
NUM_PREDICTION_TILINGS = 4
# TOTAL_FEATURE_LENGTH =NUMBER_OF_ACTIONS * (PIXEL_FEATURE_LENGTH * NUMBER_OF_PIXEL_SAMPLES + NUMBER_OF_GVFS * PREDICTION_FEATURE_LENGTH) + DID_TOUCH_FEATURE_LENGTH
TOTAL_FEATURE_LENGTH = PIXEL_FEATURE_LENGTH * NUMBER_OF_PIXEL_SAMPLES + NUMBER_OF_GVFS * PREDICTION_FEATURE_LENGTH * NUMBER_OF_ACTIONS + DID_TOUCH_FEATURE_LENGTH + 1

# Channels
RED_CHANNEL = 0
GREEN_CHANNEL = 1
BLUE_CHANNEL = 2
DEPTH_CHANNEL = 3

WALL_THRESHOLD = 0.2  # If the prediction is greater than this, the pavlov agent will avert


class StateRepresentation(object):

    def __init__(self, gvfs):
        """Initializes a representation
        Args:
          gvfs (???): a collection of GVFs.
        """
        self.gvfs = gvfs
        self.behaviorPolicy = BehaviorPolicy()
        self.pointsOfInterest = []
        self.numberOfTimesBumping = 0
        self.randomYs = np.random.choice(HEIGHT, NUMBER_OF_PIXEL_SAMPLES, replace=True)
        self.randomXs = np.random.choice(WIDTH, NUMBER_OF_PIXEL_SAMPLES, replace=True)

        for i in range(NUMBER_OF_PIXEL_SAMPLES):
            point = self.randomXs[i], self.randomYs[i]
            self.pointsOfInterest.append(point)

    def save_points_of_interest(self, file_name):
        """Saves the subsampled pixel locations to a file.
        Args:
            file_name (str): the location of the file to save points of interest to.
        """
        with open(file_name, 'wb') as outfile:
            pickle.dump(self.pointsOfInterest, outfile)

    def read_points_of_interest(self, file_name):
        """Reads the subsampled pixel locations from a
        Args:
            file_name (str): the name of the file to be read.
        """
        with open(file_name, 'rb') as inFile:
            self.pointsOfInterest = pickle.load(inFile)
            print("Read points of interest")

    def get_rgb_pixel_from_frame(self, frame, x, y):
        """Given a frame and some x,y pixel location, extracts the rgb value of the given pixel.

        Args:
            frame (???): a current minecraft frame
            x (int): the x location of the pixel of interest
            y (int): the y location of the pixel of interest

        Returns:
            (r, g, b) (tuple): the r,g,b values for the pixel of interest.
        """
        r = frame[3 * (x + y * WIDTH)]
        g = frame[1 + 3 * (x + y * WIDTH)]
        b = frame[2 + 3 * (x + y * WIDTH)]
        return (r, g, b)

    @staticmethod
    def get_empty_phi():
        return np.zeros(TOTAL_FEATURE_LENGTH)

    def get_phi(self, previous_phi, previous_action, state, simple_phi=False, ):
        """
            Name: get_phi
            Description: Creates the feature representation (phi) for a given observation. The representation
              created by individually tile coding each NUMBER_OF_PIXEL_SAMPLES rgb values together, and then assembling them.
              Finally, the didBump value is added to the end of the representation. didBump is determined to be true if
              the closest pixel in view is less than PIXEL_DISTANCE_CONSIDERED_BUMP
            Input: the observation. This is the full pixel rgbd values for each of the IMAGE_WIDTH X IMAGE_HEIGHT pixels in view
            Output: The feature vector
            """
        if simple_phi:
            return self.get_cheating_phi(state, previous_action)

        if not state:
            return None

        try:
            frame = state['visionData']
        except KeyError:
            return self.get_empty_phi()     # if there is no frame, return an empty feature vector.

        phi = []

        # For the points we are subsampling into our representation...
        for point in self.pointsOfInterest:
            # Get the pixel value at that point
            x = point[0]
            y = point[1]
            red, green, blue = self.get_rgb_pixel_from_frame(frame, x, y)
            red = red / 256.0
            green = green / 256.0
            blue = blue / 256.0

            pixel_rep = np.zeros(PIXEL_FEATURE_LENGTH)
            # Tile code these 3 values together
            indexes = tiles(NUM_IMAGE_TILINGS, PIXEL_FEATURE_LENGTH, [red, green, blue])
            pixel_rep[indexes] = 1.0
            # Assemble with other pixels
            phi.extend(pixel_rep)

        # Add the values for each of the gvf predictions + previous action using the previous state
        for name, gvf in self.gvfs.items():

            for key in self.behaviorPolicy.ACTIONS:
                prediction_rep = np.zeros(PREDICTION_FEATURE_LENGTH)
                if self.behaviorPolicy.ACTIONS[key] == previous_action:
                    prediction = gvf.prediction(previous_phi)
                    indexes = tiles(NUM_PREDICTION_TILINGS, 16, [prediction])
                    for index in indexes:
                        prediction_rep[index] = 1.0

                phi.extend(prediction_rep)

        did_touch = state['touchData']
        phi.append(float(did_touch))
        phi.append(1.0)   # bias bit
        return np.array(phi)

    def get_cheating_phi(self, state, previousAction):
        if not state:
            return None
        if len(state.video_frames) < 0:
            return self.get_empty_phi()

        phi = np.zeros(TOTAL_FEATURE_LENGTH)

        xPos = state['x']
        zPos = state['y']
        yaw = state['yaw']
        didTouch = state['touchData']

        idx = int(z) * 10 + x

        if yaw == 0:
            idx = idx + 100 * 0
        elif yaw == 90:
            idx = idx + 100 * 1
        elif yaw == 180:
            idx = idx + 100 * 2
        else:
            idx = idx + 100 * 3

        if didTouch:
            idx = idx + 400

        phi[idx] = 1
        '''
        if didTouch:
          phi[len(phi) - 1] = 1
        '''

        return phi
