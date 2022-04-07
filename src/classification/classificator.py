from src.classification.features import Sato, sobel, fast
import numpy as np


class Classificator:
    def create_feature(self, images, method):
        return [method(image)[0] for image in images]

    def __distance(self, element1, element2):
        return np.linalg.norm(np.array(element1) - np.array(element2))

    def classify(self, train, test, method, use_database=None):
        if method not in [sobel, Sato, fast]:
            return []
        if use_database:
            featured_train = use_database[method]
        else:
            featured_train = self.create_feature(train[0], method)
        featured_test = self.create_feature(test[0], method)
        answers = []

        for test_element in featured_test:
            min_element = [100000, -1]

            for i in range(len(featured_train)):
                dist = self.__distance(test_element, featured_train[i])
                if dist < min_element[0]:
                    min_element = [dist, i]

            if min_element[1] < 0:
                answers.append(0)
            else:
                answers.append(train[1][min_element[1]])

        return answers
