import numpy as np



class RBFNetwork:
    def __init__(self, k=3, tolerance=0.0001, max_iterations=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def __gaussian(self, x, c, a):
        print(x)
        print(c)
        print(a)
        print(np.exp(np.power(-np.linalg.norm(x - c), 2)))
        bias = 1
        return np.exp(np.power(-np.linalg.norm(x - c), 2)) + bias


    def __calculateOutput(self):
        output = []
        for classification in self.classes:
            x = self.classes[classification]
            c = self.centroids[classification]
            a = self.spreads[classification]
            print('Max')
            print(np.max(self.classes[classification], axis=0))
            output.append(self.__gaussian(x, c, a))
        return np.array(output)

    def fit(self, data):
        self.centroids = {}
        self.spreads = {}

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
                self.spreads[classification] = np.max(self.classes[classification], axis=0) - self.centroids[classification]

            isOptimal = True

            print(self.centroids)

            for centroid in self.centroids:

                original_centroid = previous[centroid]
                curr = self.centroids[centroid]

                print(original_centroid)
                print(curr)

                #if np.sum((curr - original_centroid) / original_centroid * 100.0) > self.tolerance:
                    #isOptimal = False

            #break out of the main loop if the results are optimal, ie. the centroids don't change their positions much(more than our tolerance)
            #if isOptimal:
                #break

            if True:
                return self.__calculateOutput()

    def pred(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

'''
def main():
    df = pd.read_csv(r"./data/ipl.csv")
    df = df[['one', 'two']]
    dataset = df.astype(float).values.tolist()

    X = df.values  # returns a numpy array

    km = RBFNetwork(3)
    km.fit(X)

    # Plotting starts here
    colors = 10 * ["r", "g", "c", "b", "k"]

    for centroid in km.centroids:
        plt.scatter(km.centroids[centroid][0], km.centroids[centroid][1], s=130, marker="x")

    for classification in km.classes:
        color = colors[classification]
        for features in km.classes[classification]:
            plt.scatter(features[0], features[1], color=color, s=30)

    plt.show()


if __name__ == "__main__":
    main()
'''