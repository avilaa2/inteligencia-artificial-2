import re
import numpy as np
import random as rn
#import tkinter as tk
from tkinter import *
from networks import RBFNetwork
from ia2 import Adaline
import math as math
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class MyDialog:
    def __init__(self, parent, info):
        self.top = Toplevel(parent)
        Label(self.top, text=info).pack()
        b = Button(self.top, text="OK", command=self.ok)
        b.pack(pady=5)

    def ok(self):
        self.top.destroy()

class InputSection(Frame):
    def __init__(self, master):
        Frame.__init__(self, master, bg='white')
        vars = {}
        vars['epochsMax'] = 300
        vars['clusters'] = 3
        vars['xPoint'] = 0.0
        vars['yPoint'] = 0.0
        vars['targetError'] = 0.000001
        vars['learningRate'] = 1.0

        print('Inicializando Variables')
        for key, value in vars.items():
            print("self.{}={}".format(key, value))
            exec("self.{}={}".format(key, value))
        self.vars = vars
        self.initUI()

    def initUI(self):
        vcmd = self.register(self.validate)


        # Options for class_list
        self.options = {
            "Coseno": 'self.cos()',
            "Seno": 'self.sin()',
            "Quadratic": 'self.quadratic()',
            "Cube": 'self.cube()',
            "2sin(x) + cos(3x)": 'self.mathFunc1()',
            "sin(2x) + ln(x^2)": 'self.mathFunc2()'
        }

        Label(self, text="Learn Rate:").grid(pady=5, row=0, column=0)
        Label(self, text="Num Epochs:").grid(pady=5, row=1, column=0)
        Label(self, text="target Error:").grid(pady=5, row=2, column=0)
        Label(self, text="Clusters:").grid(pady=5, row=0, column=2)
        Label(self, text="Functions:").grid(pady=5, row=1, column=2)

        self.key = StringVar(self)
        self.key.set('Coseno')

        self.learn_entry = Entry(self, validate="key", vcmd=(vcmd, '%P', 'float', 'learningRate'))
        self.learn_entry.grid(padx=5, row=0, column=1)
        self.epoca_entry = Entry(self, validate="key", vcmd=(vcmd, '%P', 'int', 'epochsMax'))
        self.epoca_entry.grid(padx=5, row=1, column=1)
        self.error_entry = Entry(self, validate="key", vcmd=(vcmd, '%P', 'float', 'targetError'))
        self.error_entry.grid(padx=5, row=2, column=1)
        self.clusters_entry = Entry(self, validate="key", vcmd=(vcmd, '%P', 'int', 'clusters'))
        self.clusters_entry.grid(padx=5, row=0, column=3)
        self.button_start = Button(self, text="Start")
        self.button_start.grid(padx=5, row=3, column=1)
        self.class_list = OptionMenu(self, self.key, *self.options.keys())
        self.class_list.grid(padx=5, row=1, column=3)
        self.button_train = Button(self, text="Train")
        self.button_train.grid(row=2, column=3)




    #TODO add pcolor to GUI

    def validate(self, input, type, var):
        if not input:  # the field is being cleared
            exec ("self.{}={}".format(var, vars[var]))
            return True
        try:
            print("self.{}={}('{}')".format(var, type, input))
            exec("self.{}={}('{}')".format(var, type, input))
            return True
        except ValueError:
            return False

class GraphSection(Frame):
    def __init__(self, master = None, legend = 'Default', figWidth = 8, figHeight = 8):
        self.master = master
        self.legend = legend
        self.figWith = figWidth
        self.figHeigt = figHeight
        self.initUI(master, legend, (figWidth, figHeight))


    def initUI(self, master, legend, figSize):
        Frame.__init__(self, master)
        self.line = None
        self.line2 = None
        self.legend = legend
        self.fig = plt.figure(figsize=figSize)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side='bottom', expand=False)

        if legend == 'Adaline':
            x = np.arange(-10, 10, 0.1)
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlim([-1, 1])
            self.ax.set_ylim([-10, 10])
            self.init(x, np.cos(x))
        else:
            self.ax = self.fig.add_subplot(111)
            self.init(np.arange(0, 10, 1), np.arange(0, 1, 0.1))

    def init(self, xdata, ydata):
        self.ax.set_xlim([min(xdata), max(xdata)])
        self.ax.set_ylim([min(ydata), max(ydata)])
        self.drawLine(xdata, ydata)
        self.ax.set_title(self.legend + ' Plot')
        self.ax.set_xlabel("X axis")
        self.ax.set_ylabel("Y axis")
        self.ax.legend()
        self.canvas.draw()

    def drawLine(self, x, y):
        self.ax.set_xlim([min(x), max(x)])
        self.ax.set_ylim([min(y), max(y)])
        if self.line is None:
            self.line, = self.ax.plot(x, y, label=self.legend)
        else:
            self.line.set_xdata(x)
            self.line.set_ydata(y)
        self.canvas.draw()

class MainApplication(Frame):
    def __init__(self, parent, *args, **kwargs):
        self.init()
        Frame.__init__(self, parent, *args, **kwargs)

        self.graphFrame = Frame(parent)
        self.inputSection = InputSection(parent)
        self.eGraphSection = GraphSection(self.graphFrame, 'Error', 4, 8)
        self.tGraphSection = GraphSection(self.graphFrame, 'Adaline', 8, 8)

        self.inputSection.pack(side="top")
        self.tGraphSection.pack(side="left")
        self.eGraphSection.pack(side="right")

        #self.inputSection.button_test.bind("<Button-1>", self.test)
        self.inputSection.button_start.bind("<Button-1>", self.start)
        self.inputSection.button_train.bind("<Button-1>", self.train)

        self.inputSection.pack(fill='x')
        self.graphFrame.pack()

        self.xdata = np.arange(-10, 10, 0.1)
        self.ydata = np.cos(self.xdata)

    def __drawPoint(self, id, point, color, marker):
        self.output.append(id)
        self.tGraphSection.ax.scatter(point['x'], point['y'], point['z'], c=color, marker=marker)
        self.data.append([point['x'], point['y'], point['z']])

    def cos(self):
        self.ydata = np.cos(self.xdata)

    def sin(self):
        self.ydata = np.sin(self.xdata)

    def quadratic(self):
        self.ydata = np.power(self.xdata, 2)

    def cube(self):
        self.ydata = np.power(self.xdata, 3)

    def mathFunc1(self):
        self.ydata = 2 * np.cos(self.xdata) + np.sin(3*self.xdata)

    def mathFunc2(self):
        self.ydata = np.sin(2*self.xdata) + np.log(np.power(self.xdata, 2))

    def init(self):
        self.data = []
        self.output = []

    def start(self, event):
        exec(self.inputSection.options[self.inputSection.key.get()])
        #self.tGraphSection.ax.clear()
        self.tGraphSection.drawLine(self.xdata, self.ydata)
        self.tGraphSection.canvas.draw()
        #self.tGraphSection.init(self.xdata, self.ydata)
        #self.eGraphSection.ax.clear()
        self.eGraphSection.init(np.arange(10, 0, -0.1), np.arange(0, 10, 0.1))
        self.init()

    def test(self, coord3D):
        print("hola")
        #self.mlp.xdata.append(x)
        #self.mlp.ydata.append(y)
        #result = self.mlp.test()
        #resultClass = round(result)

        #if resultClass == 0:
        #    self.__drawPoint(0, [x, y], 'ro')
        #else:
        #    self.__drawPoint(1, [x, y], 'go')

    def train(self, event):
        exec(self.inputSection.options[self.inputSection.key.get()])
        data = self.xdata

        if self.tGraphSection.line2 is not None:
            self.tGraphSection.line2.remove(0)

        # Declare RBF Network and Adaline
        self.rbfn = RBFNetwork(self.inputSection.clusters)
        self.adaline = Adaline(self.inputSection.clusters)
        self.adaline.data = self.rbfn.train(data)
        self.adaline.output = self.ydata
        self.adaline.train(self.inputSection.learningRate, self.inputSection.epochsMax, self.inputSection.targetError)

        # Plot function
        self.tGraphSection.ax.plot(self.xdata, self.adaline.z, label='RBF')
        self.tGraphSection.canvas.draw()
        self.tGraphSection.ax.lines[1].remove()

        # Plot error
        x = range(len(self.adaline.avgErrors[1:]))
        y = self.adaline.avgErrors[1:]
        self.eGraphSection.drawLine(x, y)

    def __formatCoord(self, event):
        coord3D = {}
        regex = re.compile(r'(\w)=(-?\d+\.?\d*)')
        coordString = self.tGraphSection.ax.format_coord(event.xdata, event.ydata)
        for term in coordString.split(','):
            m = regex.search(term)
            if m:
                name = m.group(1)
                value = float(m.group(2))
                coord3D[name] = value
        return coord3D

    def onclick(self, event):
        coord3D = self.__formatCoord(event)
        #print('x:{}, y:{}, z:{}'.format(event.xdata, event.ydata, event.zdata))

        try:
            if event.button == 2:
                self.test(coord3D)
            elif event.button == 1:
                self.__drawPoint(0, coord3D, 'r', 'o')
            elif event.button == 3:
                self.__drawPoint(1, coord3D, 'g', 'o')
            self.tGraphSection.canvas.draw()
        except ValueError:
            return False


root = Tk()
mainApp = MainApplication(root)
mainApp.start(True)
mainApp.pack(side="top", fill="both", expand=True)
#mainApp.tGraphSection.fig.canvas.mpl_connect('button_release_event', mainApp.onclick)
root.mainloop()