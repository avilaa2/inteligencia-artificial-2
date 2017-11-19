import re
import numpy as np
import random as rn
#import tkinter as tk
from tkinter import *
from networks import RBFNetwork
from ia2 import Adaline1D
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
        vars['layer1'] = 8
        vars['layer2'] = 4
        vars['epochsMax'] = 100
        vars['xPoint'] = 0.0
        vars['yPoint'] = 0.0
        vars['targetError'] = 0.1
        vars['learningRate'] = 0.001

        print('Inicializando Variables')
        for key, value in vars.items():
            print("self.{}={}".format(key, value))
            exec("self.{}={}".format(key, value))
        self.vars = vars
        self.initUI()

    def initUI(self):
        vcmd = self.register(self.validate)

        Label(self, text="Learn Rate:").grid(pady=5, row=0, column=0)
        Label(self, text="Num Epochs:").grid(pady=5, row=1, column=0)
        Label(self, text="target Error:").grid(pady=5, row=2, column=0)
        Label(self, text="Layer 1:").grid(pady=5, row=0, column=2)
        Label(self, text="Layer 2:").grid(pady=5, row=1, column=2)
        Label(self, text="x:").grid(pady=5, row=0, column=4)
        Label(self, text="y:").grid(pady=5, row=1, column=4)
        Label(self, text="class:").grid(pady=5, row=0, column=6)

        self.learn_entry = Entry(self, validate="key", vcmd=(vcmd, '%P', 'float', 'learningRate'))
        self.learn_entry.grid(padx=5, row=0, column=1)
        self.epoca_entry = Entry(self, validate="key", vcmd=(vcmd, '%P', 'int', 'epochsMax'))
        self.epoca_entry.grid(padx=5, row=1, column=1)
        self.error_entry = Entry(self, validate="key", vcmd=(vcmd, '%P', 'float', 'targetError'))
        self.error_entry.grid(padx=5, row=2, column=1)
        self.button_start = Button(self, text="Start")
        self.button_start.grid(padx=5, row=3, column=1)

        self.first_layer = Entry(self, validate="key", vcmd=(vcmd, '%P', 'int', 'layer1'))
        self.first_layer.grid(padx=5, row=0, column=3)
        self.second_layer = Entry(self, validate="key", vcmd=(vcmd, '%P', 'int', 'layer2'))
        self.second_layer.grid(padx=5, row=1, column=3)
        self.button_train = Button(self, text="Train")
        self.button_train.grid(row=2, column=3)

        #self.x_coordinate = Entry(self, validate="key", vcmd=(vcmd, '%P', 'float', 'xPoint'))
        #self.x_coordinate.grid(padx=5, row=0, column=5)
        #self.y_coordinate = Entry(self, validate="key", vcmd=(vcmd, '%P', 'float', 'yPoint'))
        #self.y_coordinate.grid(padx=5, row=1, column=5)
        #self.button_test = Button(self, text="Test")
        #self.button_test.grid(row=3, column=3)

        #Options for class_list
        options = {"Red": 0, "Green": 1, "Blue": 2, "Yellow": 3}
        self.class_list = OptionMenu(self, StringVar(self), *options.keys())
        self.class_list.grid(padx=5, row=0, column=7)


    #TODO add pcolor to GUI

    def validate(self, input, type, var):
        if not input:  # the field is being cleared
            exec ("self.{}={}".format(var, vars[var]))
            return True
        try:
            exec("self.{}={}('{}')".format(var, type, input))
            return True
        except ValueError:
            return False

class GraphSection(Frame):
    def __init__(self, master = None, legend = 'Default', figWidth = 8, figHeight = 8):
        self.initUI(master, legend, (figWidth, figHeight))

    def initUI(self, master, legend, figSize):
        Frame.__init__(self, master)
        self.line = None
        self.legend = legend
        self.fig = plt.figure(figsize=figSize)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side='bottom', expand=False)

        if legend == 'Adaline':
            x = np.arange(0, 1, 0.1)
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlim([-1, 1])
            self.ax.set_ylim([-10, 10])
            self.init(x, np.cos(2*np.pi*x))
        else:
            self.ax = self.fig.add_subplot(211)
            self.ax2 = self.fig.add_subplot(212)
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
        if self.line is None:
            self.line, = self.ax.plot(x, y, label=self.legend)
            print(self.line)
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

        self.adaline = Adaline1D()

    def __drawPoint(self, id, point, color, marker):
        self.output.append(id)
        self.tGraphSection.ax.scatter(point['x'], point['y'], point['z'], c=color, marker=marker)
        self.data.append([point['x'], point['y'], point['z']])

    def init(self):
        self.data = []
        self.output = []

    def start(self, event):
        xdata = np.arange(-10, 10, 0.1)
        self.tGraphSection.ax.cla()
        self.tGraphSection.init(xdata, np.cos(2*np.pi*xdata))
        self.eGraphSection.ax.cla()
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
        x = np.arange(0, 1, 0.1)
        y = np.cos(2*np.pi*x)
        self.rbfn = RBFNetwork(2)
        self.output = self.rbfn.fit(np.array([[x[i], y[i]] for i in range(len(x))]))

        print ('SALIDA:')
        print(self.output)

        for i in range(len(self.output)):
            self.adaline.xdata.append(self.output[i])
            self.adaline.output.append(y[i])
        self.adaline.train(self.inputSection.learningRate, self.inputSection.epochsMax, self.inputSection.targetError)

        X, Y, Z = axes3d.get_test_data(0.05)
        ax = self.tGraphSection.ax

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
mainApp.pack(side="top", fill="both", expand=True)
mainApp.tGraphSection.fig.canvas.mpl_connect('button_release_event', mainApp.onclick)
root.mainloop()