import random as rn
import numpy as np
#import matplotlib.pyplot as plt

class Perceptron2D:
    "Perceptron de 2 Dimensiones"

    def __init__(self):
        self.init()

    def init(self):

        self.eta = 0.1
        self.epochs = 0
        self.xdata = []
        self.ydata = []
        self.wdata = np.array([rn.random(), rn.random(), rn.random()])
        self.output = []
        self.avgErrors = []
        self.trainingSet = []
        self.done = True

    def __pw(self,x):
        "Funcion de activacion"
        if np.dot(x,self.wdata) >= 0:
            return 1
        return 0

    def arrangeData(self):
        self.output = np.array(self.output)
        N = len(self.xdata)
        for i in range(N):
            self.trainingSet.append([-1,self.xdata[i],self.ydata[i]])
        self.trainingSet = np.array(self.trainingSet)
        print(self.trainingSet)

    def recta(self,x_sample):
        "Funcion recta"
        w= self.wdata
        return -(w[1]/w[2])*x_sample + w[0]/w[2]

    def train(self,learningRate, epochsMax):
        self.arrangeData()
        self.eta = learningRate
        done = False
        while not done and self.epochs < epochsMax:
            error_count = 0
            done = True
            self.epochs += 1
            for j in range(len(self.trainingSet)):
                error = self.output[j] - self.__pw(self.trainingSet[j])
                if error != 0:
                    error_count += 1
                    done = False
                    self.wdata += self.eta * error * self.trainingSet[j]
                    print(self.wdata)
            self.avgErrors.append(error_count / 4)

    def printOutput(self):
        print("Training set and output : ")
        for j in range(len(self.trainingSet)):
            x = self.trainingSet[j]
            print(x, " : ", self.__pw(x))

class Adaline(Perceptron2D):

    def sigmoid(self, y):
        return 1/(1+ np.exp(-y))

    def train(self, learningRate, epochsMax, targetError):
        self.arrangeData()
        self.eta = learningRate
        self.avgErrors.append(1)
        while self.epochs < epochsMax and np.abs(self.avgErrors[-1]) > targetError:
            error = 0
            self.epochs += 1
            for j in range(len(self.trainingSet)):
                y = np.dot(self.trainingSet[j],self.wdata)
                fy = self.sigmoid(y)
                error_j =  self.output[j] - self.sigmoid(y)
                error += error_j
                deltaw = 2*self.eta*error_j*fy*(1-fy)*self.trainingSet[j]
                self.wdata += deltaw
                print(self.wdata)
            self.avgErrors.append(error / len(self.trainingSet))

    def verify(self):
        for j in range(len(self.trainingSet)):
            x =  np.dot(self.trainingSet[j],self.wdata)
            y = self.output[j]
            print(y, " : ", self.sigmoid(x))