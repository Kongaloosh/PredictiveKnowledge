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
    #self.root = Tk()
    self.root = Toplevel()
    self.root.wm_title("GVF Knowledge")

    self.voronoiCanvas = Canvas(self.root, borderwidth=0, highlightthickness=0, width=WIDTH, height=HEIGHT, bg="black")
    self.voronoiCanvas.grid(row = 0, column = 0)

    self.gameCanvas = Canvas(self.root, borderwidth=0, highlightthickness=0, width=WIDTH, height=HEIGHT, bg="black")
    self.gameCanvas.grid(row = 0, column = 1)

    #Did touch display
    self.didTouch = StringVar()
    self.didTouchLabel = Label(self.root, textvariable = self.didTouch, font = 'Helvetica 18 bold')
    self.didTouchLabel.grid(row = 1, columnspan = 3)


    timeStepValues = np.arange(-50, 0, 1) #The last 50

    #T
    self.tFigure = Figure(figsize=(4.5, 1.8), dpi=100)
    #self.a = self.tAFigure.add_subplot(111)
    self.tPlot = self.tFigure.add_subplot(111)
    self.tPlot.set_ylim(-0.05, 1.05)
    self.tPredictions = [0.0] * 50
    self.tPredictionLine, = self.tPlot.plot(timeStepValues, self.tPredictions, 'g', label = "T(predict)")
    self.tActualValues = [0.0] * 50
    self.tActualLine, = self.tPlot.plot(timeStepValues, self.tActualValues, 'b', label="T(actual)")

    self.tPlot.legend()
    self.tCanvas = FigureCanvasTkAgg(self.tFigure, master=self.root)

    self.tCanvas.draw()
    self.tCanvas.get_tk_widget().grid(row = 2, column = 0)


    #TL
    self.tlFigure = Figure(figsize=(4.5, 1.8), dpi=100)
    self.tlPlot = self.tlFigure.add_subplot(111)
    self.tlPlot.set_ylim(-0.05, 1.05)
    self.tlPredictions = [0.0] * 50
    self.tlPredictionLine, = self.tlPlot.plot(timeStepValues, self.tlPredictions, 'g', label = "TL(predict)")
    self.tlActualValues = [0.0] * 50
    self.tlActualLine, = self.tlPlot.plot(timeStepValues, self.tlActualValues, 'b', label="TL(actual)")

    self.tlPlot.legend()
    self.tlCanvas = FigureCanvasTkAgg(self.tlFigure, master=self.root)
    self.tlCanvas.draw()
    self.tlCanvas.get_tk_widget().grid(row = 2, column = 1)

    #TR
    self.trFigure = Figure(figsize=(4.5, 1.8), dpi=100)
    self.trPlot = self.trFigure.add_subplot(111)
    self.trPlot.set_ylim(-0.05, 1.05)
    self.trPredictions = [0.0] * 50
    self.trPredictionLine, = self.trPlot.plot(timeStepValues, self.trPredictions, 'g', label = "TR(predict)")
    self.trActualValues = [0.0] * 50
    self.trActualLine, = self.trPlot.plot(timeStepValues, self.trActualValues, 'b', label="TR(actual)")

    self.trPlot.legend()
    self.trCanvas = FigureCanvasTkAgg(self.trFigure, master=self.root)
    self.trCanvas.draw()
    self.trCanvas.get_tk_widget().grid(row = 2, column = 2)

    #TB
    self.tbFigure = Figure(figsize=(4.5, 1.8), dpi=100)
    self.tbPlot = self.tbFigure.add_subplot(111)
    self.tbPlot.set_ylim(-0.05, 1.05)
    self.tbPredictions = [0.0] * 50
    self.tbPredictionLine, = self.tbPlot.plot(timeStepValues, self.tbPredictions, 'g', label = "TB(predict)")
    self.tbActualValues = [0.0] * 50
    self.tbActualLine, = self.tbPlot.plot(timeStepValues, self.tbActualValues, 'b', label="TB(actual)")

    self.tbPlot.legend()
    self.tbCanvas = FigureCanvasTkAgg(self.tbFigure, master=self.root)
    self.tbCanvas.draw()
    self.tbCanvas.get_tk_widget().grid(row = 3, column = 0)

    #TA
    self.tAFigure = Figure(figsize=(4.5, 1.8), dpi=100)
    #self.a = self.tAFigure.add_subplot(111)
    self.taPlot = self.tAFigure.add_subplot(111)
    self.taPlot.set_ylim(-0.05, 1.05)
    self.taPredictions = [0.0] * 50
    self.taPredictionLine, = self.taPlot.plot(timeStepValues, self.taPredictions, 'g', label = "TA(predict)")
    self.taActualValues = [0.0] * 50
    self.taActualLine, = self.taPlot.plot(timeStepValues, self.taActualValues, 'b', label="TA(actual)")

    self.taPlot.legend()
    self.taCanvas = FigureCanvasTkAgg(self.tAFigure, master=self.root) #canvas.get_tk_widget().grid(row=1,column=4,columnspan=3,rowspan=20)

    self.taCanvas.draw()
    #self.taCanvas.get_tk_widget().pack(side = "top", anchor = "w")
    self.taCanvas.get_tk_widget().grid(row = 3, column = 1)

    #DTA
    self.dtaFigure = Figure(figsize=(4.5, 1.8), dpi=100)
    self.dtaPlot = self.dtaFigure.add_subplot(111)
    self.dtaPlot.set_ylim(-1, 12)
    self.dtaPredictions = [0.0] * 50
    self.dtaPredictionLine, = self.dtaPlot.plot(timeStepValues, self.dtaPredictions, 'g', label = "DTA(predict)")
    self.dtaActualValues = [0.0] * 50
    self.dtaActualLine, = self.dtaPlot.plot(timeStepValues, self.dtaActualValues, 'b', label="DTA(actual)")

    self.dtaPlot.legend()
    self.dtaCanvas = FigureCanvasTkAgg(self.dtaFigure, master=self.root)
    self.dtaCanvas.draw()
    self.dtaCanvas.get_tk_widget().grid(row = 3, column = 2)

    #DTL
    self.dtlFigure = Figure(figsize=(4.5, 1.8), dpi=100)
    self.dtlPlot = self.dtlFigure.add_subplot(111)
    self.dtlPlot.set_ylim(-1, 12)
    self.dtlPredictions = [0.0] * 50
    self.dtlPredictionLine, = self.dtlPlot.plot(timeStepValues, self.dtlPredictions, 'g', label = "DTL(predict)")
    self.dtlActualValues = [0.0] * 50
    self.dtlActualLine, = self.dtlPlot.plot(timeStepValues, self.dtlActualValues, 'b', label="DTL(actual)")

    self.dtlPlot.legend()
    self.dtlCanvas = FigureCanvasTkAgg(self.dtlFigure, master=self.root)
    self.dtlCanvas.draw()
    self.dtlCanvas.get_tk_widget().grid(row = 4, column = 0)


    #DTR
    self.dtrFigure = Figure(figsize=(4.5, 1.8), dpi=100)
    self.dtrPlot = self.dtrFigure.add_subplot(111)
    self.dtrPlot.set_ylim(-1, 12)
    self.dtrPredictions = [0.0] * 50
    self.dtrPredictionLine, = self.dtrPlot.plot(timeStepValues, self.dtrPredictions, 'g', label = "DTR(predict)")
    self.dtrActualValues = [0.0] * 50
    self.dtrActualLine, = self.dtrPlot.plot(timeStepValues, self.dtrActualValues, 'b', label="DTR(actual)")

    self.dtrPlot.legend()
    self.dtrCanvas = FigureCanvasTkAgg(self.dtrFigure, master=self.root)
    self.dtrCanvas.draw()
    self.dtrCanvas.get_tk_widget().grid(row = 4, column = 1)

    #DTB
    self.dtbFigure = Figure(figsize=(4.5, 1.8), dpi=100)
    self.dtbPlot = self.dtbFigure.add_subplot(111)
    self.dtbPlot.set_ylim(-1, 12)
    self.dtbPredictions = [0.0] * 50
    self.dtbPredictionLine, = self.dtbPlot.plot(timeStepValues, self.dtbPredictions, 'g', label = "DTB(predict)")
    self.dtbActualValues = [0.0] * 50
    self.dtbActualLine, = self.dtbPlot.plot(timeStepValues, self.dtbActualValues, 'b', label="DTB(actual)")

    self.dtbPlot.legend()
    self.dtbCanvas = FigureCanvasTkAgg(self.dtbFigure, master=self.root)
    self.dtbCanvas.draw()
    self.dtbCanvas.get_tk_widget().grid(row = 4, column = 2)

    #DLF
    self.wlfFigure = Figure(figsize=(4.5, 1.8), dpi=100)
    self.wlfPlot = self.wlfFigure.add_subplot(111)
    self.wlfPlot.set_ylim(-1, 11)
    self.wlfPredictions = [0.0] * 50
    self.wlfPredictionLine, = self.wlfPlot.plot(timeStepValues, self.wlfPredictions, 'g', label = "WLF(predict)")
    self.wlfActualValues = [0.0] * 50
    self.wlfActualLine, = self.wlfPlot.plot(timeStepValues, self.wlfActualValues, 'b', label="WLF(actual)")

    self.wlfPlot.legend()
    self.wlfCanvas = FigureCanvasTkAgg(self.wlfFigure, master=self.root)
    self.wlfCanvas.draw()
    self.wlfCanvas.get_tk_widget().grid(row = 5, column = 0)

    #Number of steps
    self.numberOfSteps = StringVar()
    self.numberOfStepsLabel = Label(self.root, textvariable = self.numberOfSteps)
    self.numberOfStepsLabel.grid(row = 6, columnspan = 3)
    #self.numberOfStepsLabel.pack(side = "top", anchor = "w")


    self.reset()



  def reset(self):
    self.voronoiCanvas.delete("all")

    self.voronoiImage = Image.new('RGB', (WIDTH, HEIGHT))
    self.voronoiPhotoImage = None
    self.voronoiImage_handle = None
    self.current_frame = 0

  def update(self, voronoiImage,
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
             wallAdjacent,
             wallLeftForward,
             wallLeftForwardPrediction):

    #Update Steps
    self.numberOfSteps.set("Step: " + str(numberOfSteps))

    #Update did touch
    if didTouch:
      self.didTouch.set("TOUCHED")
    else:
      self.didTouch.set("")

    #Update game image
    #change from BGR to RGB
    l = len(voronoiImage)
    voronoiImage = cv2.cvtColor(voronoiImage, cv2.COLOR_BGR2RGB)
    # convert the cv2 images to PIL format...
    self.voronoiImage = Image.fromarray(voronoiImage)

    # ...and then to ImageTk format
    self.voronoiPhotoImage = ImageTk.PhotoImage(self.voronoiImage)


    # And update/create the canvas image:
    if self.voronoiImage_handle is None:
      self.voronoiImage_handle = self.voronoiCanvas.create_image(WIDTH/2,HEIGHT/2,
                                               image=self.voronoiPhotoImage)
    else:
      self.voronoiCanvas.itemconfig(self.voronoiImage_handle, image=self.voronoiPhotoImage)

    #Update plots

    #T
    self.tPredictions.pop(0)
    self.tPredictions.append(currentTouchPrediction)
    self.tActualValues.pop(0)
    if (wallInFront):
      touchActual = 1.0
    else:
      touchActual = 0.0
    self.tActualValues.append(touchActual)
    self.tPredictionLine.set_ydata(self.tPredictions)
    self.tActualLine.set_ydata(self.tActualValues)
    self.tCanvas.draw()

    #TL
    self.tlPredictions.pop(0)
    self.tlPredictions.append(turnLeftAndTouchPrediction)
    self.tlActualValues.pop(0)
    if (wallOnLeft):
      touchActual = 1.0
    else:
      touchActual = 0.0
    self.tlActualValues.append(touchActual)
    self.tlPredictionLine.set_ydata(self.tlPredictions)
    self.tlActualLine.set_ydata(self.tlActualValues)
    self.tlCanvas.draw()

    #TR
    self.trPredictions.pop(0)
    self.trPredictions.append(turnRightAndTouchPrediction)
    self.trActualValues.pop(0)
    if (wallOnRight):
      touchActual = 1.0
    else:
      touchActual = 0.0
    self.trActualValues.append(touchActual)
    self.trPredictionLine.set_ydata(self.trPredictions)
    self.trActualLine.set_ydata(self.trActualValues)
    self.trCanvas.draw()

    #TB
    self.tbPredictions.pop(0)
    self.tbPredictions.append(touchBehindPrediction)
    self.tbActualValues.pop(0)
    if (wallBehind):
      touchActual = 1.0
    else:
      touchActual = 0.0
    self.tbActualValues.append(touchActual)
    self.tbPredictionLine.set_ydata(self.tbPredictions)
    self.tbActualLine.set_ydata(self.tbActualValues)
    self.tbCanvas.draw()

    #TA
    self.taPredictions.pop(0)
    self.taPredictions.append(touchAdjacentPrediction)
    self.taActualValues.pop(0)
    if (wallAdjacent):
      touchActual = 1.0
    else:
      touchActual = 0.0
    self.taActualValues.append(wallAdjacent)
    self.taPredictionLine.set_ydata(self.taPredictions)
    self.taActualLine.set_ydata(self.taActualValues)
    self.taCanvas.draw()

    #DTA
    self.dtaPredictions.pop(0)
    self.dtaPredictions.append(distanceToAdjacentPrediction)
    self.dtaActualValues.pop(0)
    self.dtaActualValues.append(distanceToAdjacent)
    self.dtaPredictionLine.set_ydata(self.dtaPredictions)
    self.dtaActualLine.set_ydata(self.dtaActualValues)
    self.dtaCanvas.draw()

    #DTL
    self.dtlPredictions.pop(0)
    self.dtlPredictions.append(distanceToLeftPrediction)
    self.dtlActualValues.pop(0)
    self.dtlActualValues.append(distanceToLeft)
    self.dtlPredictionLine.set_ydata(self.dtlPredictions)
    self.dtlActualLine.set_ydata(self.dtlActualValues)
    self.dtlCanvas.draw()

    #DTR
    self.dtrPredictions.pop(0)
    self.dtrPredictions.append(distanceToRightPrediction)
    self.dtrActualValues.pop(0)
    self.dtrActualValues.append(distanceToRight)
    self.dtrPredictionLine.set_ydata(self.dtrPredictions)
    self.dtrActualLine.set_ydata(self.dtrActualValues)
    self.dtrCanvas.draw()

    #DTB
    self.dtbPredictions.pop(0)
    self.dtbPredictions.append(distanceBackPrediction)
    self.dtbActualValues.pop(0)
    self.dtbActualValues.append(distanceBack)
    self.dtbPredictionLine.set_ydata(self.dtbPredictions)
    self.dtbActualLine.set_ydata(self.dtbActualValues)
    self.dtbCanvas.draw()

    #WLF
    self.wlfPredictions.pop(0)
    self.wlfPredictions.append(wallLeftForwardPrediction)
    self.wlfActualValues.pop(0)
    self.wlfActualValues.append(wallLeftForward)
    self.wlfPredictionLine.set_ydata(self.wlfPredictions)
    self.wlfActualLine.set_ydata(self.wlfActualValues)
    self.wlfCanvas.draw()

    self.root.update()

