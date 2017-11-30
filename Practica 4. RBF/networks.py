import numpy as np



class RBFNetwork:
    def __init__(self, k=3, tolerance=0.0001, max_iterations=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.spreads = {}

    def __gaussian(self, x, d, sigma):
        r = np.linalg.norm(x - d)
        return np.exp( -(np.power(r, 2)) / (2 * np.power(sigma, 2)))

    def test(self, x):
        self.result = []
        for i in range(len(x)):
            output = []
            for classification in self.classes:
                d = self.centroids[classification]
                sigma = self.spreads[classification]
                output.append(self.__gaussian(x[i], d, sigma))
            self.result.append(output)
        return self.result


    def train(self, data):
        self.centroids = {}

        # initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
        for i in range(self.k):
            self.centroids[i] = data[i]

        # begin iterations
        for i in range(self.max_iterations):
            self.classes = {}
            for i in range(self.k):
                self.classes[i] = []

            # find the distance between the point and cluster; choose the nearest centroid
            for features in data:
                distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(features)

            previous = dict(self.centroids)

            # average the cluster datapoints to re-calculate the centroids
            for classification in self.classes:
                self.centroids[classification] = np.average(self.classes[classification], axis=0)

            # break out of the main loop if the results are optimal, ie. the centroids don't change their positions much(more than our tolerance)
            if np.equal(previous, self.centroids):
                break

        # Set Spreads
        for i in self.classes:
            distances = []
            for j in self.classes:
                if i != j:
                    distances += [np.linalg.norm(self.centroids[i] - self.centroids[j])]
            self.spreads[i] = min(distances)

        for classification in self.classes:
            print('class: {} , data: {}'.format(classification, self.classes[classification]))

        return self.test(data)

    def pred(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification