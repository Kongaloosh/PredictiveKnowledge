
from __future__ import print_function

from builtins import range
import os
import sys
import time
import numpy as np
import json
import cv2
from Voronoi import *
from constants import *
from StateRepresentation import *
import peakAtState as peak
from BehaviorPolicy import *
from display import *
from GVF import *
from GridWorld import *
from PIL import ImageTk
from PIL import Image

if sys.version_info[0] == 2:
  sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
  import functools

  print = functools.partial(print, flush=True)


def didTouchCumulant(phi):
  if USE_SIMPLE_PHI:

    idx = np.nonzero(phi)[0][0]
    if (idx) < 400:
      return 0.0
    else:
      return 1.0
  else:
    return phi[len(phi) - 1]



class Foreground:

  def __init__(self):
    self.gridWorld = GridWorld('model/grids', initialX=1, initialY = 1)
    self.behaviorPolicy = BehaviorPolicy()

    self.display = Display()
    self.gvfs = {}
    self.configureGVFs(simplePhi=USE_SIMPLE_PHI)
    self.stateRepresentation = StateRepresentation(self.gvfs)
    self.state = False
    self.oldState = False
    self.phi = self.stateRepresentation.getEmptyPhi()
    self.oldPhi = self.stateRepresentation.getEmptyPhi()

  def configureGVFs(self, simplePhi=False):
    touchThreshold = 0.8  # The prediction value before it is considered to be true.
    '''
    Layer 1 - Touch (T)
    '''
    touchGVF = None
    if simplePhi:

      touchGVF = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH, alpha=0.70,
                     isOffPolicy=True, name="T")
    else:
      touchGVF = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH,
                     alpha=0.10 / (NUM_IMAGE_TILINGS * NUMBER_OF_PIXEL_SAMPLES), isOffPolicy=True, name="T")

    touchGVF.cumulant = didTouchCumulant
    touchGVF.policy = self.behaviorPolicy.extendHandPolicy

    def didtouchGamma(phi):
      return 0

    touchGVF.gamma = didtouchGamma
    self.gvfs[touchGVF.name] = touchGVF

    '''
    #Layer 2 - Touch Left (TL) and Touch Right (TR)
    '''
    if simplePhi:
      turnLeftGVF = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH,
                        alpha=0.70, isOffPolicy=True, name="TL")
    else:
      turnLeftGVF = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH,
                        alpha=0.10 / (NUM_IMAGE_TILINGS * NUMBER_OF_PIXEL_SAMPLES), isOffPolicy=True, name="TL")

    turnLeftGVF.cumulant = self.gvfs['T'].prediction
    turnLeftGVF.policy = self.behaviorPolicy.turnLeftPolicy
    turnLeftGVF.gamma = didtouchGamma
    self.gvfs[turnLeftGVF.name] = turnLeftGVF

    if simplePhi:
      turnRightGVF = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH,
                         alpha=0.70, isOffPolicy=True, name="TR")
    else:
      turnRightGVF = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH,
                         alpha=0.10 / (NUM_IMAGE_TILINGS * NUMBER_OF_PIXEL_SAMPLES), isOffPolicy=True, name="TR")

    turnRightGVF.cumulant = self.gvfs['T'].prediction
    turnRightGVF.policy = self.behaviorPolicy.turnRightPolicy
    turnRightGVF.gamma = didtouchGamma
    self.gvfs[turnRightGVF.name] = turnRightGVF

    '''
    #Layer 3 - Touch Behind
    '''
    if simplePhi:
      touchBehindGVF = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH,
                        alpha=0.70, isOffPolicy=True, name="TB")
    else:
      touchBehindGVF = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH,
                        alpha=0.10 / (NUM_IMAGE_TILINGS * NUMBER_OF_PIXEL_SAMPLES), isOffPolicy=True, name="TB")

    touchBehindGVF.cumulant = self.gvfs['TR'].prediction
    touchBehindGVF.policy = self.behaviorPolicy.turnRightPolicy
    touchBehindGVF.gamma = didtouchGamma
    self.gvfs[touchBehindGVF.name] = touchBehindGVF


    '''
    #Layer 4 - Touch Adjacent (TA)
    '''

    if simplePhi:
      touchAdjacentGVF = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH,
                             alpha=0.70, isOffPolicy=True, name="TA")
    else:

      touchAdjacentGVF = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH,
                             alpha=0.10 / (NUM_IMAGE_TILINGS * NUMBER_OF_PIXEL_SAMPLES), isOffPolicy=True, name="TA")
                             

    def touchAdjacentCumulant(phi):

      touchAdjacent = 0.0
      touchAdjacent = max([self.gvfs['T'].prediction(phi), self.gvfs['TL'].prediction(phi), self.gvfs['TR'].prediction(phi), self.gvfs['TB'].prediction(phi)])

      if touchAdjacent > touchThreshold:
        touchAdjacent = 1.0
      else:
        touchAdjacent = 0.0

      return touchAdjacent

    touchAdjacentGVF.cumulant = touchAdjacentCumulant
    touchAdjacentGVF.policy = self.behaviorPolicy.turnRightPolicy

    def touchAdjacentGama(phi):
      if touchAdjacentCumulant(phi) == 1.0:
        return 0
      else:
        return 1


    touchAdjacentGVF.gamma = touchAdjacentGama
    self.gvfs[touchAdjacentGVF.name] = touchAdjacentGVF

    '''
    #Layer 5 - Distance to touch adjacent (DTA)
    Measures how many steps the agent is from being adjacent touch something.
    * Note that because our agent only rotates 90 degrees at a time, this is basically the 
     number of steps to a wall. So the cumulant could be T. But we have the cumulant as TA instead 
     since this would allow for an agent whose rotations are not 90 degrees.
    '''

    if simplePhi:
      distanceToTouchAdjacentGVF = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH,
                             alpha=0.70, isOffPolicy=True, name="DTA")
    else:

      distanceToTouchAdjacentGVF = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH,
                             alpha=0.10 / (NUM_IMAGE_TILINGS * NUMBER_OF_PIXEL_SAMPLES), isOffPolicy=True, name="DTA")

    def distanceToTouchAdjacentCumulant(phi):
      return 1.0

    distanceToTouchAdjacentGVF.cumulant = distanceToTouchAdjacentCumulant
    distanceToTouchAdjacentGVF.policy = self.behaviorPolicy.moveForwardPolicy

    def distanceToTouchAdjacentGamma(phi):
      prediction = self.gvfs['T'].prediction(phi) #TODO - change to self.gvfs['TA'].prediction() after testing
      if prediction > touchThreshold:
        return 0
      else:
        return 1

    distanceToTouchAdjacentGVF.gamma = distanceToTouchAdjacentGamma

    self.gvfs[distanceToTouchAdjacentGVF.name] = distanceToTouchAdjacentGVF


    '''
    Layer 6 - Distance to Left (DTL), distance to right (DTR), distance back (DTB)
    Measures how many steps to the left, or right, or behind,the agent is from a wall.
    '''

    #Distance to Left GVF
    if simplePhi:
      distanceToLeftGVF = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH,
                             alpha=0.70, isOffPolicy=True, name="DTL")
    else:

      distanceToLeftGVF = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH,
                             alpha=0.10 / (NUM_IMAGE_TILINGS * NUMBER_OF_PIXEL_SAMPLES), isOffPolicy=True, name="DTL")

    def distanceToLeftCumulant(phi):
      return self.gvfs['DTA'].prediction(phi)

    distanceToLeftGVF.cumulant = distanceToLeftCumulant

    def distanceToLeftGamma(phi):
      return 0

    distanceToLeftGVF.gamma = distanceToLeftGamma
    distanceToLeftGVF.policy = self.behaviorPolicy.turnLeftPolicy
    self.gvfs[distanceToLeftGVF.name] = distanceToLeftGVF

    # Distance to Right GVF
    if simplePhi:
      distanceToRightGVF = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH,
                              alpha=0.70, isOffPolicy=True, name="DTR")
    else:

      distanceToRightGVF = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH,
                              alpha=0.10 / (NUM_IMAGE_TILINGS * NUMBER_OF_PIXEL_SAMPLES), isOffPolicy=True, name="DTR")

    def distanceToRightCumulant(phi):
      return self.gvfs['DTA'].prediction(phi)

    distanceToRightGVF.cumulant = distanceToRightCumulant

    def distanceToRightGamma(phi):
      return 0

    distanceToRightGVF.gamma = distanceToRightGamma
    distanceToRightGVF.policy = self.behaviorPolicy.turnRightPolicy
    self.gvfs[distanceToRightGVF.name] = distanceToRightGVF

    # Distance behind GVF
    if simplePhi:
      distanceToBackGVF = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH,
                              alpha=0.70, isOffPolicy=True, name="DTB")
    else:

      distanceToBackGVF = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH,
                              alpha=0.10 / (NUM_IMAGE_TILINGS * NUMBER_OF_PIXEL_SAMPLES), isOffPolicy=True, name="DTB")

    def distanceToBackCumulant(phi):
      return self.gvfs['DTR'].prediction(phi)

    distanceToBackGVF.cumulant = distanceToBackCumulant

    def distanceToBackGamma(phi):
      return 0

    distanceToBackGVF.gamma = distanceToBackGamma
    distanceToBackGVF.policy = self.behaviorPolicy.turnRightPolicy
    self.gvfs[distanceToBackGVF.name] = distanceToBackGVF


  def learn(self):
    for name, gvf in self.gvfs.items():
      gvf.learn(lastState=self.oldPhi, action=self.action, newState=self.phi)

  def updateUI(self):
    # Create a voronoi image

    frameError = False
    try:
      frame = self.state['visionData']
    except:
      frameError = True
      print("Error gettnig frame")

    if not frameError:
      # rgb = self.s
      voronoi = voronoi_from_pixels(pixels=frame, dimensions=(WIDTH, HEIGHT),
                                    pixelsOfInterest=self.stateRepresentation.pointsOfInterest)
      # cv2.imshow('My Image', voronoi)
      # cv2.waitKey(0)

      if self.oldState == False:
        didTouch = False
      else:
        didTouch = self.oldState['touchData']

      inFront = peak.isWallInFront(self.state['x'], self.state['y'], self.state['yaw'], self.gridWorld)
      touchPrediction = self.gvfs['T'].prediction(self.phi)


      '''
      #For debugging
      previousInFront = peak.isWallInFront(self.oldState['x'], self.oldState['y'], self.oldState['yaw'], self.gridWorld)
      previousTouchPrediction = self.gvfs['T'].prediction(self.oldPhi)

      if not previousInFront and previousTouchPrediction > 0.0:
        print("Bad first learning. ")
        print("Last action: " + self.action)
        msg = self.oldState.observations[0].text
        observations = json.loads(msg)  # and parse the JSON

        yaw = observations.get(u'Yaw', 0)
        x = observations.get(u'XPos', 0)
        z = observations.get(u'ZPos', 0)
        print("From: " + str(yaw) + ", " + str(x) + ", " + str(z))

        msg = self.state.observations[0].text
        observations = json.loads(msg)  # and parse the JSON
        yaw = observations.get(u'Yaw', 0)
        x = observations.get(u'XPos', 0)
        z = observations.get(u'ZPos', 0)
        print("To: " + str(yaw) + ", " + str(x) + ", " + str(z))
        ph = self.stateRepresentation.getPhi(previousPhi = self.oldPhi, state=self.state, previousAction=self.action, simplePhi = USE_SIMPLE_PHI)
        idx = np.nonzero(ph)[0][0]
        numNonZeros = len(np.nonzero(ph)[0])
        print("idx: " + str(idx) + ", nonZeros: " + str(numNonZeros))
        print("Observations since last:" + str(self.state.number_of_observations_since_last_state))
        print("")
      '''
      onLeft = peak.isWallOnLeft(self.state['x'], self.state['y'], self.state['yaw'], self.gridWorld)
      turnLeftAndTouchPrediction = self.gvfs['TL'].prediction(self.phi)

      onRight = peak.isWallOnRight(self.state['x'], self.state['y'], self.state['yaw'], self.gridWorld)
      turnRightAndtouchPrediction = self.gvfs['TR'].prediction(self.phi)

      isBehind = peak.isWallBehind(self.state['x'], self.state['y'], self.state['yaw'], self.gridWorld)
      touchBehindPrediction = self.gvfs['TB'].prediction(self.phi)

      wallAdjacent = peak.isWallAdjacent(self.state['x'], self.state['y'], self.state['yaw'], self.gridWorld)
      isWallAdjacentPrediction = self.gvfs['TA'].prediction(self.phi)

      distanceToAdjacent = peak.distanceToAdjacent(self.state['x'], self.state['y'], self.state['yaw'], self.gridWorld)
      distanceToAdjacentPrediction = self.gvfs['DTA'].prediction(self.phi)

      distanceLeft = peak.distanceLeftToAdjacent(self.state['x'], self.state['y'], self.state['yaw'], self.gridWorld)
      distanceLeftPrediction = self.gvfs['DTL'].prediction(self.phi)

      distanceRight = peak.distanceRightToAdjacent(self.state['x'], self.state['y'], self.state['yaw'], self.gridWorld)
      distanceRightPrediction = self.gvfs['DTR'].prediction(self.phi)

      distanceBack = peak.distanceBehindToAdjacent(self.state['x'], self.state['y'], self.state['yaw'], self.gridWorld)
      distanceBackPrediction = self.gvfs['DTB'].prediction(self.phi)

      self.display.update(image=voronoi,
                          numberOfSteps=self.actionCount,
                          currentTouchPrediction=touchPrediction,
                          wallInFront=inFront,
                          didTouch=didTouch,
                          turnLeftAndTouchPrediction=turnLeftAndTouchPrediction,
                          wallOnLeft=onLeft,
                          turnRightAndTouchPrediction=turnRightAndtouchPrediction,
                          touchBehindPrediction = touchBehindPrediction,
                          wallBehind = isBehind,
                          touchAdjacentPrediction=isWallAdjacentPrediction,
                          wallAdjacent=wallAdjacent,
                          wallOnRight=onRight,
                          distanceToAdjacent = distanceToAdjacent,
                          distanceToAdjacentPrediction = distanceToAdjacentPrediction,
                          distanceToLeft = distanceLeft,
                          distanceToLeftPrediction = distanceLeftPrediction,
                          distanceToRight = distanceRight,
                          distanceToRightPrediction = distanceRightPrediction,
                          distanceBack = distanceBack,
                          distanceBackPrediction = distanceBackPrediction
                          )
      # time.sleep(1.0)



  def start(self):

    self.actionCount = 0

    self.action = self.behaviorPolicy.ACTIONS['extend_hand']

    # Loop until mission ends:
    while True:
      self.actionCount += 1
      # print(".", end="")

      self.oldState = self.state
      self.oldPhi = self.phi

      # Select and send action. Need to sleep to give time for simulator to respond
      self.action = self.behaviorPolicy.mostlyForwardAndTouchPolicy(self.state)
      observation = self.gridWorld.takeAction(self.action)
      #time.sleep(0.2)
      self.state = observation

      print("==========")
      print("Action was: " + str(self.action))
      # print("Number of observations since last: " + str(self.state.number_of_observations_since_last_state))
      # print("Length of observation array: " + str(len(self.state.observations)))
      # print("Number of video frames: " + str(len(self.state.video_frames)))
      if self.oldState:
        yaw = self.oldState['yaw']
        xPos = self.oldState['x']
        zPos = self.oldState['y']
        print("From observation: (" + str(xPos) + ", " + str(zPos) + "), yaw:" + str(yaw))

      yaw = self.state['yaw']
      xPos = self.state['x']
      zPos = self.state['y']
      print("To observation: (" + str(xPos) + ", " + str(zPos) + "), yaw:" + str(yaw))
      """
      if (self.action == "turn -1"):
        #Debug the video
        i = 0
        for videoframe in self.state.video_frames:
          cmap = Image.frombytes('RGB', (WIDTH, HEIGHT), bytes(videoframe.pixels))
          cmap.show(title = "Image: " + str(i))
          i+=1
        i = i
      """
      print("")

      self.phi = self.stateRepresentation.getPhi(previousPhi=self.oldPhi, state=self.state,
                                                 previousAction=self.action, simplePhi=USE_SIMPLE_PHI)

      # Do the learning
      self.learn()

      # Update our display (for debugging and progress reporting)
      self.updateUI()

    print()
    print("Mission ended")
    # Mission has ended.


fg = Foreground()
fg.start()