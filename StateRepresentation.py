
#!/usr/bin/env python


from constants import *

import numpy as np
from tiles import *
import json
import time
from BehaviorPolicy import *

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

NUMBER_OF_COLOR_CHANNELS = 3 #red, blue, green
PIXEL_FEATURE_LENGTH = np.power(NUM_IMAGE_INTERVALS, NUMBER_OF_COLOR_CHANNELS) * NUM_IMAGE_TILINGS
PREDICTION_FEATURE_LENGTH = 16
DID_TOUCH_FEATURE_LENGTH = 1
NUMBER_OF_GVFS = 9
NUMBER_OF_ACTIONS = 4
NUM_PREDICTION_TILINGS = 4
#TOTAL_FEATURE_LENGTH =NUMBER_OF_ACTIONS * (PIXEL_FEATURE_LENGTH * NUMBER_OF_PIXEL_SAMPLES + NUMBER_OF_GVFS * PREDICTION_FEATURE_LENGTH) + DID_TOUCH_FEATURE_LENGTH
TOTAL_FEATURE_LENGTH =PIXEL_FEATURE_LENGTH * NUMBER_OF_PIXEL_SAMPLES + NUMBER_OF_GVFS * PREDICTION_FEATURE_LENGTH * NUMBER_OF_ACTIONS  + DID_TOUCH_FEATURE_LENGTH

# Channels
RED_CHANNEL = 0
GREEN_CHANNEL = 1
BLUE_CHANNEL = 2
DEPTH_CHANNEL = 3

WALL_THRESHOLD = 0.2 #If the prediction is greater than this, the pavlov agent will avert

class StateRepresentation(object):
  def __init__(self, gvfs):
    self.gvfs = gvfs
    self.behaviorPolicy = BehaviorPolicy()
    self.pointsOfInterest = []
    self.numberOfTimesBumping = 0
    self.randomYs = np.random.choice(HEIGHT, NUMBER_OF_PIXEL_SAMPLES, replace=True)
    self.randomXs = np.random.choice(WIDTH, NUMBER_OF_PIXEL_SAMPLES, replace=True)

    for i in range(NUMBER_OF_PIXEL_SAMPLES):
      point = self.randomXs[i], self.randomYs[i]
      self.pointsOfInterest.append(point)


  def getRGBPixelFromFrame(self, frame, x, y):
    length = len(frame)
    r = frame[3 * (x + y * WIDTH)]
    g = frame[1 + 3 * (x + y * WIDTH)]
    b = frame[2 + 3 * (x + y * WIDTH)]
    return (r, g, b)


  def getEmptyPhi(self):
    return np.zeros(TOTAL_FEATURE_LENGTH)

  """
  Name: getPhi
  Description: Creates the feature representation (phi) for a given observation. The representation
    created by individually tile coding each NUMBER_OF_PIXEL_SAMPLES rgb values together, and then assembling them. 
    Finally, the didBump value is added to the end of the representation. didBump is determined to be true if
    the closest pixel in view is less than PIXEL_DISTANCE_CONSIDERED_BUMP
  Input: the observation. This is the full pixel rgbd values for each of the IMAGE_WIDTH X IMAGE_HEIGHT pixels in view
  Output: The feature vector
  """
  def getPhi(self, previousPhi, previousAction, state, simplePhi = False, ):
    if simplePhi:
      return self.getCheatingPhi(state, previousAction)

    if not state:
      return None

    try:
      frame = state['visionData']
    except:
      return self.getEmptyPhi()

    phi = []

    for point in self.pointsOfInterest:
      #Get the pixel value at that point
      x = point[0]
      y = point[1]
      red, green, blue = self.getRGBPixelFromFrame(frame, x, y)
      red = red / 256.0
      green = green / 256.0
      blue = blue / 256.0

      pixelRep = np.zeros(PIXEL_FEATURE_LENGTH)
      #Tile code these 3 values together
      indexes = tiles(NUM_IMAGE_TILINGS, PIXEL_FEATURE_LENGTH, [red, green, blue])
      for index in indexes:
        pixelRep[index] = 1.0

      #Assemble with other pixels
      phi.extend(pixelRep)

    #Add the values for each of the gvf predictions + previous action using the previous state
    for name, gvf in self.gvfs.items():
      for key in self.behaviorPolicy.ACTIONS:
        predictionRep = np.zeros(PREDICTION_FEATURE_LENGTH)
        if self.behaviorPolicy.ACTIONS[key] == previousAction:
          prediction = gvf.prediction(previousPhi)
          indexes = tiles(NUM_PREDICTION_TILINGS, 16, [prediction])
          for index in indexes:
            predictionRep[index] = 1.0

        phi.extend(predictionRep)

    didTouch = state['touchData']
    phi.append(int(didTouch))

    return np.array(phi)


  def getCheatingPhi(self, state, previousAction):
    if not state:
      return None
    if len(state.video_frames) < 0:
      return self.getEmptyPhi()


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
      idx = idx + 100 *2
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

