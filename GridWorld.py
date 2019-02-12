from random import randint
import numpy as np
import random
from PIL import ImageTk
from PIL import Image
import json
from constants import *
import pickle
from Voronoi import *

class GridWorld:

  def __init__(self, modelFile, initialX = 0, initialY = 0, initialYaw = 0):
    #Initialize the environment using the configuration file
    print("Initializing grid world ...")
    self.grids = {}
    with open(modelFile, 'rb') as gridDataFile:
      self.grids = pickle.load(gridDataFile)

    self.currentX = initialX
    self.currentY = initialY
    self.currentYaw = initialYaw
    self.ACTIONS = [
      'forward',
      'turn_left',
      'turn_right',
      'extend_hand'
    ]

    #self.imageDictionary = {}
    print("Initiailized grid world")

  """
  def imageDataForFile(self, imageFile):
    #Queries memory for the data to return it. Reads from disk and stores to memory if not available.
    if imageFile in self.imageDictionary:
      return self.imageDictionary[imageFile]
    else:
      with open(imageFile, "rb") as imageFile:
        f = imageFile.read()
        b = bytearray(f)
        self.imageDictionary[imageFile] = b
        return b

  """
  def gridFor(self, x, y):
    k = self.keyNameFor(x, y)
    if k in self.grids:
      return self.grids[k]
    else:
      return None

  def keyNameFor(self, x, y):
    return str(x) + ',' + str(y)



  def takeAction(self, action):

    #Set the new orientation
    if action not in self.ACTIONS:
      print("Error. specified action not in action set")
      return

    didTouch = False

    if action == 'turn_left':
      self.currentYaw = (self.currentYaw - 90) % 360
    elif action == 'turn_right':
      self.currentYaw = (self.currentYaw + 90) % 360
    elif action == 'extend_hand':
      desiredKey = ''
      if self.currentYaw == 0:
        desiredKey = self.keyNameFor(self.currentX, self.currentY + 1)
      elif self.currentYaw == 90:
        desiredKey = self.keyNameFor(self.currentX - 1, self.currentY)
      elif self.currentYaw == 180:
        desiredKey = self.keyNameFor(self.currentX, self.currentY - 1)
      elif self.currentYaw == 270:
        desiredKey = self.keyNameFor(self.currentX + 1, self.currentY)

      if not desiredKey in self.grids:
        didTouch = True

    elif action == 'forward':
      if self.currentYaw == 0:
        desiredKey = self.keyNameFor(self.currentX, self.currentY + 1)
        if desiredKey in self.grids:
          #Move south
          self.currentY = self.currentY + 1
      elif self.currentYaw == 90:
        desiredKey = self.keyNameFor(self.currentX - 1, self.currentY)
        if desiredKey in self.grids:
          #Move west
          self.currentX = self.currentX - 1
      elif self.currentYaw == 180:
        desiredKey = self.keyNameFor(self.currentX, self.currentY - 1)
        if desiredKey in self.grids:
          #move north
          self.currentY = self.currentY - 1
      elif self.currentYaw == 270:
        desiredKey = self.keyNameFor(self.currentX + 1, self.currentY)
        if desiredKey in self.grids:
          #move east
          self.currentX = self.currentX + 1

    currentGridKey = self.keyNameFor(self.currentX, self.currentY)

    #pixelData = self.imageDataForFile(self.grids[currentGridKey][str(self.currentYaw)])
    pixelData = self.grids[currentGridKey][str(self.currentYaw)]
    return {'visionData': pixelData, 'touchData': didTouch, 'reward':0, 'x': self.currentX, 'y': self.currentY, 'yaw': self.currentYaw}


##########################################
## testing ###############################
##########################################

"""
def printImageFromObs(obs, pointsOfInterest):
  voronoi = voronoi_from_pixels(pixels=obs['visionData'], dimensions=(WIDTH, HEIGHT),
                                pixelsOfInterest=pointsOfInterest)
  cv2.imshow('My Image', voronoi)
  cv2.waitKey(0)

randomYs = np.random.choice(HEIGHT, NUMBER_OF_PIXEL_SAMPLES, replace=True)
randomXs = np.random.choice(WIDTH, NUMBER_OF_PIXEL_SAMPLES, replace=True)
pointsOfInterest = []

for i in range(100):
  point = randomXs[i], randomYs[i]
  pointsOfInterest.append(point)

gridWorld = GridWorld('model/grids', initialX=1, initialY = 1)
obs = gridWorld.takeAction('forward')
printImageFromObs(obs, pointsOfInterest)

obs = gridWorld.takeAction('turn_left')
printImageFromObs(obs, pointsOfInterest)

obs = gridWorld.takeAction('turn_right')
printImageFromObs(obs, pointsOfInterest)

obs = gridWorld.takeAction('extend_hand')
printImageFromObs(obs, pointsOfInterest)

"""