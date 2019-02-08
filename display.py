from constants import *
import sys
import cv2
if sys.version_info[0] == 2:
  # Workaround for https://github.com/PythonCharmers/python-future/issues/262
  from Tkinter import *
else:
  from tkinter import *

from PIL import ImageTk
from PIL import Image


import matplotlib, sys
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


video_width = 432
video_height = 240

DISPLAY_WIDTH = WIDTH
DISPLAY_HEIGHT = 432 + video_height

class Display(object):
  def __init__(self):
    plt.ion() #turn matplot interactive on
    self.root = Tk()
    self.root.wm_title("GVF Knowledge")


    #Data 1
    self.tAFigure = Figure(figsize=(4.3,2), dpi=100)
    #self.a = self.tAFigure.add_subplot(111)
    self.taPlot = self.tAFigure.add_subplot(111)
    self.taPlot.set_ylim(-0.05, 1.05)
    timeStepValues = np.arange(-50, 0, 1) #The last 50
    self.taPredictions = [0.0] * 50
    self.taPredictionLine, = self.taPlot.plot(timeStepValues, self.taPredictions, 'g', label = "TA(predict)")
    self.taActualValues = [0.0] * 50
    self.taActualLine, = self.taPlot.plot(timeStepValues, self.taActualValues, 'b', label="TA(actual)")

    self.taPlot.legend()
    self.taCanvas = FigureCanvasTkAgg(self.tAFigure, master=self.root) #canvas.get_tk_widget().grid(row=1,column=4,columnspan=3,rowspan=20)

    self.taCanvas.draw()
    self.taCanvas.get_tk_widget().pack(side = "top", anchor = "w")




    self.canvas = Canvas(self.root, borderwidth=0, highlightthickness=0, width=WIDTH, height=HEIGHT, bg="black")
    #self.canvas.config(width=WIDTH, height=HEIGHT)
    self.canvas.pack(padx=0, pady=0)
    #self.root_frame.pack()

    #Did touch display
    self.didTouch = StringVar()
    self.didTouchLabel = Label(self.root, textvariable = self.didTouch, font = 'Helvetica 18 bold')
    self.didTouchLabel.pack()

    #Touch prediction
    self.touchPrediction = StringVar()
    self.touchPredictionLabel = Label(self.root, textvariable = self.touchPrediction)
    self.touchPredictionLabel.pack(side = "top", anchor = "w")

    #Turn left and touch prediction
    self.turnLeftAndTouchPrediction = StringVar()
    self.turnLeftAndTouchPredictionLabel = Label(self.root, textvariable = self.turnLeftAndTouchPrediction)
    self.turnLeftAndTouchPredictionLabel.pack(side = "top", anchor = "w")

    #Turn right and touch prediction
    self.turnRightAndTouchPrediction = StringVar()
    self.turnRightAndTouchPredictionLabel = Label(self.root, textvariable = self.turnRightAndTouchPrediction)
    self.turnRightAndTouchPredictionLabel.pack(side = "top", anchor = "w")

    #Touch behind prediction
    self.touchBehindPrediction = StringVar()
    self.touchBehindPredictionLabel = Label(self.root, textvariable = self.touchBehindPrediction)
    self.touchBehindPredictionLabel.pack(side = "top", anchor = "w")


    #Wall Adjacent prediction
    self.isWallAdjacentPrediction = StringVar()
    self.isWallAdjacentPredictionLabel = Label(self.root, textvariable = self.isWallAdjacentPrediction)
    self.isWallAdjacentPredictionLabel.pack(side = "top", anchor = "w")

    #Distance to adjacent prediction
    self.distanceToAdjacent = StringVar()
    self.distanceToAdjacentLabel = Label(self.root, textvariable = self.distanceToAdjacent)
    self.distanceToAdjacentLabel.pack(side = "top", anchor = "w")

    #Number of Steps Left
    self.numberOfStepsLeft = StringVar()
    self.numberOfStepsLeftLabel = Label(self.root, textvariable=self.numberOfStepsLeft)
    self.numberOfStepsLeftLabel.pack(side="top", anchor="w")

    #Number of Steps Right
    self.numberOfStepsRight = StringVar()
    self.numberOfStepsRightLabel = Label(self.root, textvariable=self.numberOfStepsRight)
    self.numberOfStepsRightLabel.pack(side="top", anchor="w")

    #Number of Steps Back
    self.numberOfStepsBack = StringVar()
    self.numberOfStepsBackLabel = Label(self.root, textvariable=self.numberOfStepsBack)
    self.numberOfStepsBackLabel.pack(side="top", anchor="w")

    #Number of steps
    self.numberOfSteps = StringVar()
    self.numberOfStepsLabel = Label(self.root, textvariable = self.numberOfSteps)
    self.numberOfStepsLabel.pack(side = "top", anchor = "w")

    self.reset()



  def reset(self):
    self.canvas.delete("all")

    self.image = Image.new('RGB', (WIDTH, HEIGHT))
    self.photoImage = None
    self.image_handle = None
    self.current_frame = 0

  def update(self, image,
             numberOfSteps,
             currentTouchPrediction,
             didTouch,
             turnLeftAndTouchPrediction,
             wallInFront,
             wallOnLeft,
             turnRightAndTouchPrediction,
             wallOnRight,
             touchBehindPrediction,
             wallBehind,
             touchAdjacentPrediction,
             distanceToAdjacent,
             distanceToAdjacentPrediction,
             distanceToLeft,
             distanceToLeftPrediction,
             distanceToRight,
             distanceToRightPrediction,
             distanceBack,
             distanceBackPrediction,
             wallAdjacent):

    #Update labels
    self.touchPrediction.set("T: " + str(currentTouchPrediction))
    if wallInFront:
      self.touchPredictionLabel.config(fg = 'blue')
    else:
      self.touchPredictionLabel.config(fg = 'red')

    self.turnLeftAndTouchPrediction.set("TL: " + str(turnLeftAndTouchPrediction))
    if wallOnLeft:
      self.turnLeftAndTouchPredictionLabel.config(fg='blue')
    else:
      self.turnLeftAndTouchPredictionLabel.config(fg = 'red')

    self.turnRightAndTouchPrediction.set("TR: " + str(turnRightAndTouchPrediction))
    if wallOnRight:
      self.turnRightAndTouchPredictionLabel.config(fg='blue')
    else:
      self.turnRightAndTouchPredictionLabel.config(fg='red')

    self.touchBehindPrediction.set("TB: " + str(touchBehindPrediction))
    if wallBehind:
      self.touchBehindPredictionLabel.config(fg='blue')
    else:
      self.touchBehindPredictionLabel.config(fg='red')

    self.isWallAdjacentPrediction.set("TA: " + str(touchAdjacentPrediction))
    if wallAdjacent:
      self.isWallAdjacentPredictionLabel.config(fg = 'blue')
    else:
      self.isWallAdjacentPredictionLabel.config(fg = 'red')

    self.distanceToAdjacent.set("DTA: " + str(round(distanceToAdjacentPrediction, 1)) + " (" + str(distanceToAdjacent) + ")")
    self.numberOfStepsLeft.set("DTL: " + str(round(distanceToLeftPrediction, 1)) + " (" + str(distanceToLeft) + ")")
    self.numberOfStepsRight.set("DTR: " + str(round(distanceToRightPrediction, 1)) + " (" + str(distanceToRight) + ")")
    self.numberOfStepsBack.set("DTB: " + str(round(distanceBackPrediction)) + " (" + str(distanceBack) + ")")

    self.numberOfSteps.set("Step: " + str(numberOfSteps))
    if didTouch:
      self.didTouch.set("TOUCHED")
    else:
      self.didTouch.set("")

    #Update image
    #change from BGR to RGB
    l = len(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # convert the cv2 images to PIL format...
    self.image = Image.fromarray(image)

    # ...and then to ImageTk format
    self.photoImage = ImageTk.PhotoImage(self.image)


    # And update/create the canvas image:
    if self.image_handle is None:
      self.image_handle = self.canvas.create_image(WIDTH/2,HEIGHT/2,
                                               image=self.photoImage)
    else:
      self.canvas.itemconfig(self.image_handle, image=self.photoImage)

    self.taPredictions.pop(0)
    self.taPredictions.append(currentTouchPrediction)
    self.taActualValues.pop(0)
    if (wallInFront):
      touchActual = 1.0
    else:
      touchActual = 0.0
    self.taActualValues.append(touchActual)
    self.taPredictionLine.set_ydata(self.taPredictions)
    self.taActualLine.set_ydata(self.taActualValues)
    self.taCanvas.draw()
    self.root.update()

