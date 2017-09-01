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
        self.__trainingSet = []
        self.done = True

    def __pw(self,x):
        "Funcion de activacion"
        if np.dot(x,self.wdata) >= 0:
            return 1
        return 0

    def __arrangeData(self):
        self.output = np.array(self.output)
        N = len(self.xdata)
        for i in range(N):
            self.__trainingSet.append([-1,self.xdata[i],self.ydata[i]])
        self.__trainingSet = np.array(self.__trainingSet)
        print(self.__trainingSet)

    def recta(self,x_sample):
        "Funcion recta"
        w= self.wdata
        return -(w[1]/w[2])*x_sample + w[0]/w[2]

    def rectaRand(self,x_sample):
        "Funcion recta"
        self.wdata = np.array([rn.random(), rn.random(), rn.random()])
        return self.recta(x_sample)

    def train(self, learningRate, epochsMax):
        self.__arrangeData()
        self.eta = learningRate
        self.done = False
        while not self.done and self.epochs < epochsMax:
            error_count = 0
            self.done = True
            self.epochs += 1
            for j in range(len(self.__trainingSet)):
                error = self.output[j] - self.__pw(self.__trainingSet[j])
                if error != 0:
                    error_count += 1
                    self.done = False
                    self.wdata += self.eta * error * self.__trainingSet[j]
                    print(self.wdata)
            self.avgErrors.append(error_count / len(self.__trainingSet))

    def printOutput(self):
        print("Training set and output : ")
        for j in range(len(self.__trainingSet)):
            x = self.__trainingSet[j]
            print(x, " : ", self.__pw(x))

'''
#Ejemplo------------------------------------------------------
p = Perceptron2D()

p.xdata = [3,6,4,1]
p.ydata = [4,1,1,2]
p.output = [1,1,0,0]

p.train()

#plot
plt.plot([3,6],[4,1],'ro')
plt.plot([4,1],[1,2],'bo')
xdata = np.arange(-6,6,0.1)
plt.plot(xdata,p.recta(xdata))
plt.show()

#Resultados
print("Iteracion: ", p.epochs)
print("Pesos: ", p.wdata)
print("Errores: ", p.avgErrors)
p.printOutput()
w = p.wdata
print("pendiente m:", -(w[1]/w[2]))
print("cruce",w[0]/w[2])
print("Ecuacion de la recta :", "y=",-(w[1]/w[2]),"x+",w[0]/w[2])
'''



