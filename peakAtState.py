import json
from GridWorld import *

# For debugging - tells us if there is indeed
def isWallInFront(currentX, currentY, yaw,  gridWorld):
  yaw = yaw % 360
  isBlock = False
  desiredX = currentX
  desiredY = currentY
  yaw = yaw % 360
  if (yaw == 0.0):
    # Facing south
    desiredY = desiredY + 1
  elif (yaw == 90.0):
    # Facing west
    desiredX = desiredX - 1
  elif (yaw == 180.0):
    # Facing north
    desiredY = desiredY - 1
  elif (yaw == -90 or yaw == 270):
    # Facing east
    desiredX = desiredX + 1

  grid = gridWorld.gridFor(desiredX, desiredY)
  if not grid == None:
    isBlock = True

  return isBlock

def isWallOnLeft(currentX, currentY, yaw,  gridWorld):
  return isWallInFront(currentX, currentY, yaw - 90, gridWorld)

def isWallOnRight(currentX, currentY, yaw,  gridWorld):
  return isWallInFront(currentX, currentY, yaw + 90, gridWorld)

def isWallBehind(currentX, currentY, yaw,  gridWorld):
  return isWallInFront(currentX, currentY, yaw + 180, gridWorld)

def isWallAdjacent(currentX, currentY, yaw, gridWorld):
  return isWallOnLeft(currentX, currentY, yaw, gridWorld) or isWallOnRight(currentX, currentY, yaw, gridWorld) or isWallInFront(currentX, currentY, yaw, gridWorld) or isWallBehind(currentX, currentY, yaw, gridWorld)


def distanceToAdjacent(currentX, currentY, yaw, gridWorld):
  yaw = yaw % 360
  if isWallInFront(currentState, gridWorld):
    atWall = True
    return 1
  else:
    if (yaw == 0.0):
      # Facing south
      return 1 + distanceToAdjacent(x, currentY + 1, yaw, gridWorld)
    elif (yaw == 90.0):
      # Facing west
      return 1 + distanceToAdjacent(x - 1, currentY, yaw, gridWorld)
    elif (yaw == 180.0):
      # Facing north
      return 1 + distanceToAdjacent(x, currentY - 1, yaw, gridWorld)
    elif (yaw == -90 or yaw == 270):
      # Facing east
      return 1 + distanceToAdjacent(x + 1, currentY, yaw, gridWorld)




def distanceLeftToAdjacent(currentX, currentY, yaw, gridWorld):
  return distanceToAdjacent(currentX, currentY, yaw - 90, gridWorld)

def distanceRightToAdjacent(currentState):
  return distanceToAdjacent(currentX, currentY, yaw + 90, gridWorld)

def distanceBehindToAdjacent(currentState):
  return distanceToAdjacent(currentX, currentY, yaw + 180, gridWorld)
