#!/usr/bin/python

import cv2
import numpy as np
import random

from PIL import ImageTk
from PIL import Image

from constants import *
from StateRepresentation import *

stateRepresentation = StateRepresentation(gvfs = None)
"""
Description:
This function creates a voronoi image representation of the supplied image, using only the information at the provided
pixel indexes.

The function creates an empty canvas with the dimensions of the provided pixels
Input: 
pixels: pixel byte array
pixelsOfInterest: tuple of format (x, y values . These are the sampled pixels/values for which the voronoi should be 
based off of
dimensions: tuple (width, height) of desired image.
Output:
The cv2 formatted voronoi image 
"""
def voronoi_from_pixels(pixels, dimensions, pixelsOfInterest):

  #Create the subdiv which will hold the voronoi information
  rect = (0, 0, dimensions[0], dimensions[1])
  subdiv = cv2.Subdiv2D(rect);
  for pixel in pixelsOfInterest:
    p = pixel[0], pixel[1]
    subdiv.insert(p)

  # Allocate space for Voronoi Diagram
  #img_voronoi should have shape (height, width, channels)
  img_voronoi = np.zeros(shape=(HEIGHT, WIDTH, 3), dtype=np.uint8)

  # Draw Voronoi diagram

  (facets, centers) = subdiv.getVoronoiFacetList([])
  facetCount = len(facets)
  for i in range(0, len(facets)):
    ifacet_arr = []
    for f in facets[i]:
      ifacet_arr.append(f)

    ifacet = np.array(ifacet_arr, np.int)

    p = centers[i]
    y = int(p[1])
    x = int(p[0])

    rgb = stateRepresentation.getRGBPixelFromFrame(pixels, x, y)
    #Get BGR color
    color = [rgb[2], rgb[1], rgb[0]]

    cv2.fillConvexPoly(img_voronoi, ifacet, color, cv2.LINE_AA, 0);
    ifacets = np.array([ifacet])
    cv2.polylines(img_voronoi, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)

  # Show results
  #For debugging - can show now
  #cv2.imshow("Vornoi Image", img_voronoi)
  #cv2.waitKey(0)

  s = img_voronoi.shape
  l = len(img_voronoi)
  return img_voronoi

# Draw voronoi diagram
def draw_voronoi(original, img_canvas, subdiv):
  (facets, centers) = subdiv.getVoronoiFacetList([])
  facetCount = len(facets)
  for i in range(0, len(facets)):
    ifacet_arr = []
    for f in facets[i]:
      ifacet_arr.append(f)

    ifacet = np.array(ifacet_arr, np.int)

    p = centers[i]
    x = int(p[1])
    y = int(p[0])
    originalPixel = original[x,y]

    color = originalPixel.tolist()

    cv2.fillConvexPoly(img_canvas, ifacet, color, cv2.LINE_AA, 0);
    ifacets = np.array([ifacet])
    cv2.polylines(img_canvas, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)
    #cv2.circle(img_canvas, (centers[i][0], centers[i][1]), 3, (0, 0, 0), 2, cv2.LINE_AA, 0)

    return img_canvas

if __name__ == '__main__':
  print("Do stuff")