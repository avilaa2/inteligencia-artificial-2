import random as rn
import numpy as np
#import matplotlib.pyplot as plt

class Perceptron:
    "Perceptron de 2 Dimensiones"

    def __init__(self, k):
        self.init(k)

    def init(self, k):
        self.k = k
        self.eta = 0.1
        self.epochs = 0
        self.data = []
        self.wdata =  np.array([rn.random() for _ in range(k)])
        self.output = []
        self.avgErrors = []
        self.trainingSet = []
        self.done = True
        self.net = 0
        self.sensitiv = 0
        self.error = 0
        self.spreads = []
        self.centroids = []

    def __pw(self,x):
        "Funcion de activacion"
        return np.dot(x,self.wdata)
        '''if np.dot(x,self.wdata) >= 0:
            return 1
        return 0'''

    def arrangeData(self):
        self.trainingSet = []
        self.output = np.array(self.output)
        self.trainingSet = np.array(self.data)

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
            self.avgErrors.append(error_count / 4)

class Adaline(Perceptron):

    def setWeights(self, len):
        """Metodo para crear dinamicamente los pesos"""
        self.wdata = []
        for i in range(len):
            self.wdata.append(rn.random())
        self.wdata = np.array(self.wdata)

    def sigmoid(self, y):
        return 1/(1+ np.exp(-y))

    def train(self, learningRate, epochsMax, targetError):
        mse = 1000000
        self.arrangeData()
        self.eta = learningRate
        self.avgErrors.append(1)
        while self.epochs < epochsMax and mse > targetError:
            mse = 0
            print('entered')
            self.z = []
            self.epochs += 1
            for k in range(len(self.trainingSet)):
                z = np.dot(self.wdata, self.trainingSet[k]) + 1
                self.z.append(z)
                error = self.output[k] - z
                mse += error
                deltaw = self.eta * error * self.trainingSet[k];
                self.wdata += deltaw
            mse = mse*mse/2;
            self.avgErrors.append(mse)
            print('Error: {}, Target: {}'.format(mse, targetError))
        #self.verify()

    def verify(self):
        self.z = []
        for j in range(len(self.trainingSet)):
            z =  np.dot(self.trainingSet[j],self.wdata)
            y = self.output[j]
            self.z.append(z)
            print('Deseada: {} ,  Obtenida: {}'.format(y, z))