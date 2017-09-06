import numpy as np
import random as rn
import tkinter as tk
from ia2 import Adaline
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class MyDialog:
    def __init__(self, parent, info):
        self.top = tk.Toplevel(parent)
        tk.Label(self.top, text=info).pack()
        b = tk.Button(self.top, text="OK", command=self.ok)
        b.pack(pady=5)

    def ok(self):
        self.top.destroy()

class InputSection(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master, bg='white')
        vcmd = self.register(self.validate)

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.columnconfigure(3, weight=1)
        self.columnconfigure(4, weight=1)
        self.columnconfigure(5, weight=1)
        self.columnconfigure(6, weight=1)
        self.columnconfigure(7, weight=1)
        self.columnconfigure(8, weight=3)
        self.columnconfigure(9, weight=1)
        self.columnconfigure(10, weight=1)
        self.columnconfigure(11, weight=1)

        self.epochsMax = 100
        self.targetError = 0.001
        self.learningRate = 0.1

        tk.Label(self, bg="white", text="Learn Rate").grid(row=0, column=0, sticky='e')
        tk.Label(self, bg="white", text="Num Epocas").grid(row=1, column=0, sticky='e')
        tk.Label(self, bg="white", text="Error deseado").grid(row=2, column=0, sticky='e')

        self.learn_entry = tk.Entry(self, bg="white", bd=2, validate="key", validatecommand=(vcmd, '%P', 'float', 'self.learningRate'))
        self.learn_entry.grid(row=0, column=1)

        self.epoca_entry = tk.Entry(self, bg="white", bd=2, validate="key", validatecommand=(vcmd, '%P', 'int', 'self.epochsMax'))
        self.epoca_entry.grid(row=1, column=1)

        self.error_entry = tk.Entry(self, bg="white", bd=2, validate="key",validatecommand=(vcmd, '%P', 'float', 'self.targetError'))
        self.error_entry.grid(row=2, column=1)

        self.button_start = tk.Button(self, text="Start")
        self.button_start.grid(row=3, column=1)

        self.button_train = tk.Button(self, text="Train")
        self.button_train.grid(row=0, column=8, rowspan=3)

    def validate(self, input, type, var):
        if not input:  # the field is being cleared
            self.epochsMax = 100
            self.learningRate = 0.1
            self.targetError = 0.001
            return True

        try:
            exec(var + " = " + type + "('" + input + "')")
            return True
        except ValueError:
            return False

class GraphSection(tk.Frame):
    def __init__(self, master, legend):
        tk.Frame.__init__(self, master)
        self.legend = legend
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side='bottom', expand=False)
        #self.canvas.show()

        if legend == 'Error':
            self.init(np.arange(0, 10), np.arange(0, 1, 0.1))
        else:
            self.init(np.arange(-10, 10, 0.1), np.arange(-10, 10, 0.1))

    def init(self, xdata, ydata):
        self.line, = self.ax.plot(xdata, ydata, label=self.legend)
        self.ax.set_title(self.legend + ' Plot')
        self.ax.set_xlabel("X axis")
        self.ax.set_ylabel("Y axis")
        self.ax.legend()
        self.canvas.draw()

class MainApplication(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        self.init()
        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.graphFrame = tk.Frame(parent)
        self.inputSection = InputSection(parent)
        self.eGraphSection = GraphSection(self.graphFrame, 'Error')
        self.tGraphSection = GraphSection(self.graphFrame, 'Adaline')

        self.inputSection.pack(side="top")
        self.tGraphSection.pack(side="left")
        self.eGraphSection.pack(side="right")

        self.inputSection.button_start.bind("<Button-1>", self.start)
        self.inputSection.button_train.bind("<Button-1>", self.train)

        self.inputSection.pack(fill='x')
        self.graphFrame.pack()

        self.adaline = Adaline()

    def init(self):
        self.data = []
        self.output = []

    def start(self, event):
        xdata = np.arange(-10, 10, 0.1)

        self.tGraphSection.ax.cla()
        self.tGraphSection.init(xdata, self.adaline.rectaRand(xdata))

        self.eGraphSection.ax.cla()
        self.eGraphSection.init(np.arange(0, 10), np.arange(0, 10))

        self.init()


    def train(self, event):
        self.adaline.wdata = np.array([rn.random(), rn.random(), rn.random()])

        for i in range(len(self.data)):
            self.adaline.xdata.append(self.data[i][0])
            self.adaline.ydata.append(self.data[i][1])
            self.adaline.output.append(self.output[i])
        self.adaline.train(self.inputSection.learningRate, self.inputSection.epochsMax, self.inputSection.targetError)

        xdata = np.arange(-10, 10, 0.1)
        self.tGraphSection.line.set_xdata(xdata)
        self.tGraphSection.line.set_ydata(self.adaline.recta(xdata))
        self.tGraphSection.canvas.draw()

        for i in self.adaline.avgErrors:
            print(i)

        self.eGraphSection.ax.set_xlim([0, len(self.adaline.avgErrors)])
        self.eGraphSection.ax.set_ylim([0, np.max(self.adaline.avgErrors)])
        self.eGraphSection.line.set_xdata([i for i in range(len(self.adaline.avgErrors))])
        self.eGraphSection.line.set_ydata(self.adaline.avgErrors)
        self.eGraphSection.canvas.draw()

        self.adaline.verify()
        '''
        if self.adaline.done == True:
            MyDialog(self.graphFrame, 'Es linealmente separable')
        else:
            MyDialog(self.graphFrame, 'No es linealmente separable')
        '''


        self.adaline.init()


    def onclick(self, event):
        try:
            print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                (event.button, event.x, event.y, event.xdata, event.ydata))
        except ValueError:
            return False

        if event.button == 1:
            self.output.append(0)
            self.tGraphSection.ax.plot(event.xdata, event.ydata, 'ro')
            self.data.append([event.xdata, event.ydata])
        elif event.button == 3:
            self.output.append(1)
            self.tGraphSection.ax.plot(event.xdata, event.ydata, 'go')
            self.data.append([event.xdata, event.ydata])

        self.tGraphSection.canvas.draw()

root = tk.Tk()
mainApp = MainApplication(root)
mainApp.pack(side="top", fill="both", expand=True)
mainApp.tGraphSection.fig.canvas.mpl_connect('button_press_event', mainApp.onclick)
root.mainloop()