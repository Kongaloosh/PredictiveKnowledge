from random import randint
import numpy as np
import random

class BehaviorPolicy:

  def __init__(self):
    self.lastAction = 0
    self.i = 0

    self.ACTIONS = {
      'forward': "forward",
      'turn_left': "turn_left",
      'turn_right': "turn_right",
      'extend_hand':"extend_hand"
    }
    '''
    self.ACTIONS = {
      'forward': "move 1",
      'back': "move -1",
      'turn_left': "turn -1",
      'turn_right': "turn 1",
      'extend_hand':"move 0",
      'no_move': "move 0"
    }
    '''

  def policy(self, state):
    self.i = self.i + 1

    isFacingWall = state[len(state) - 1] == 1 #Last bit in the feature representation represents facing the wall
    if isFacingWall:
      return self.ACTIONS['look_left']
    else:
      return self.ACTIONS['forward']

  def randomTurnPolicy(self, state):
    moves = [self.ACTIONS['turn_left'], self.ACTIONS['turn_right']]
    return moves[randint(0,1)]
  def forwardThenLeftPolicy(self,state):
    self.i+=1
    if self.i%20 == 0:
      return self.turnLeftPolicy(state)
    else:
      return self.moveForwardPolicy(state)

  def mostlyForwardPolicy(self,state):
    if self.i %21 == 0:
      return self.randomPolicy(state)
    else:
      return self.moveForwardPolicy(state)


  def mostlyForwardAndTouchPolicy(self, state):
    self.i += 1

    if self.i % 50 == 0:
      return self.turnRightPolicy(state)
    elif (self.i -1) % 50 == 0:
      return self.turnRightPolicy(state)
    elif ((self.i -2) %50) == 0:
      return self.turnRightPolicy(state)
    elif self.i % 7 == 0:
      return self.randomTurnPolicy(state)
    elif self.i % 8 == 0:
      return self.randomTurnPolicy(state)
    elif self.i % 19 ==0:
      return self.randomPolicy(state)
    elif self.i % 21 == 0:
      return self.mostlyForwardPolicy(state)
    elif self.i % 23 == 0:
      return self.mostlyForwardPolicy(state)
    else:
      if self.i % 2 == 0 and self.i < 30000:
        return self.ACTIONS['extend_hand']
      elif (self.i - 1 ) % 4 == 0:
        return self.randomPolicy(state)
      else:
        return self.mostlyForwardPolicy(state)


  def extendHandPolicy(self, state):
    return self.ACTIONS['extend_hand']

  def randomPolicy(self, state):
    return self.ACTIONS[random.choice(list(self.ACTIONS.keys()))]

  def moveForwardPolicy(self, state):
    return self.ACTIONS['forward']

  def turnLeftPolicy(self, state):
    return self.ACTIONS['turn_left']

  def turnRightPolicy(self, state):
    return self.ACTIONS['turn_right']

  def epsilonGreedyPolicy(self, state):
    print("Do something here")