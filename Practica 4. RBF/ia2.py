import random as rn
import numpy as np
#import matplotlib.pyplot as plt

class Perceptron1D:
    "Perceptron de 2 Dimensiones"

    def __init__(self):
        self.init()

    def init(self):

        self.eta = 0.1
        self.epochs = 0
        self.xdata = []
        self.wdata = np.array([rn.random(), rn.random()])
        self.output = []
        self.avgErrors = []
        self.trainingSet = []
        self.done = True
        self.net = 0
        self.sensitiv = 0
        self.error = 0

    def __pw(self,x):
        "Funcion de activacion"
        if np.dot(x,self.wdata) >= 0:
            return 1
        return 0

    def arrangeData(self):
        self.trainingSet = []
        self.output = np.array(self.output)
        N = len(self.xdata)
        for i in range(N):
            self.trainingSet.append([-1,self.xdata[i]])
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
                self.error = self.output[j] - self.__pw(self.trainingSet[j])
                if self.error != 0:
                    error_count += 1
                    done = False
                    self.wdata += self.eta * self.error * self.trainingSet[j]
                    print(self.wdata)
            self.avgErrors.append(error_count / 4)

    def printOutput(self):
        print("Training set and output : ")
        for j in range(len(self.trainingSet)):
            x = self.trainingSet[j]
            print(x, " : ", self.__pw(x))

class Adaline1D(Perceptron1D):

    def setWeights(self, len):
        """Metodo para crear dinamicamente los pesos"""
        self.wdata = []
        for i in range(len):
            self.wdata.append(rn.random())
        self.wdata = np.array(self.wdata)

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
        self.verify()

    def verify(self):
        for j in range(len(self.trainingSet)):
            x =  np.dot(self.trainingSet[j],self.wdata)
            y = self.output[j]
            print(y, " : ", self.sigmoid(x))

class MLP:

    def __init__(self,layers = []):
        """ Inicia el MLP a partir de un vector con los tamaños de las capas, exluyendo capa de entrada ej. [8 4 1]"""
        self.eta = 0.1
        self.epochs = 0
        self.xdata = []
        self.ydata = []
        self.output = []
        self.avgErrors = []
        self.trainingSet = []
        self.a = []
        self.sensitiv = []
        self.layers = []
        for i in range(len(layers)):
            self.layers.append([])
            for j in range(layers[i]):
                self.layers[i].append(Adaline())
                if(i > 0):
                    self.layers[i][j].setWeights(layers[i-1]+1)
                else:
                    self.layers[i][j].setWeights(3)

    def arrangeData(self):
        self.trainingSet = []
        self.output = np.array(self.output)
        N = len(self.xdata)
        for i in range(N):
            self.trainingSet.append([-1,self.xdata[i],self.ydata[i]])
        self.trainingSet = np.array(self.trainingSet)
        #print(self.trainingSet)

    def forward(self, a0):
        """Calcula las net y vectores de entrada a[]"""
        self.a = []
        self.a.append(a0)
        self.sensitiv = []
        for i in range(len(self.layers)):
            self.a.append([-1])
            self.sensitiv.append([])
            for j in range(len(self.layers[i])):
                neuron = self.layers[i][j]
                neuron.net = np.dot(self.a[i], neuron.wdata)
                self.a[i + 1].append(neuron.sigmoid(neuron.net))

    def backward(self, numEj):
        """Calcula sesitividad (no tomar en cuenta los umbrales w0)"""
        for i in range(len(self.layers)-1, -1, -1):
            if i == len(self.layers)-1:
                self.sensitiv[i] = -2*np.diag(self.getNetDerivVector(i)).dot(self.output[numEj] - self.a[i+1][1])
            else:
                self.sensitiv[i] = np.diag(self.getNetDerivVector(i)).dot(self.getWeightMatrixWithout(i + 1).transpose())
                self.sensitiv[i] = self.sensitiv[i].dot(self.sensitiv[i + 1])

    def test(self):
        self.arrangeData()
        self.forward(self.trainingSet[-1])
        print('Resultado: {}'.format(self.a[-1][1]))
        return self.a[-1][1]

    def train(self, learningRate, epochsMax, targetError):
        self.arrangeData()
        self.eta = learningRate
        self.avgErrors.append(1)

        while self.epochs < epochsMax and self.avgErrors[-1] > targetError:
            error = 0
            mse = 0
            self.epochs += 1
            for j in range(len(self.trainingSet)):
                self.forward(self.trainingSet[j])
                self.backward(j)
                self.updateWeights()
                error = self.output[j] - self.a[len(self.a)-1][1]
                mse = mse + pow(error,2)
            self.avgErrors.append(mse)
        print(self.avgErrors[-1])

    def getNetDerivVector(self, posLayer):
        netVector = []
        for i in range(len(self.layers[posLayer])):
            fy = self.layers[posLayer][i].sigmoid(self.layers[posLayer][i].net)
            netVector.append(fy * (1 - fy))
        return netVector

    def getWeightMatrixWithout(self, posLayer):
        weightVector = ([])
        for i in range(len(self.layers[posLayer])):
            weightVector.append(self.layers[posLayer][i].wdata[1:len(self.layers[posLayer][i].wdata)])
        weightVector = np.array(weightVector)
        return weightVector

    def updateWeights(self):
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                neuron = self.layers[i][j]
                neuron.wdata += -self.eta * self.sensitiv[i][j] * self.a[i]

    def rectas(self, x_sample):
        rectas = []
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                neuron = self.layers[i][j]
                rectas.append(neuron.recta(x_sample))
        return rectas

    def verify(self):
        for i in range(len(self.trainingSet)):
            self.forward(self.trainingSet[i])
            print(self.output[i], " : ", self.a[-1][1])

    '''
    def verify(self, x):
        self.forward(x)
        output = self.a[-1][1]
        print(output)
        return output
    '''