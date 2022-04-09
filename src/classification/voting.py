from matplotlib import pyplot as plt
from src.classification.features import *
from src.classification.constants import *
from src.classification.classificator import Classificator


def voting(train, test, SHOW=False, use_database=None):
    result = {}
    voted_answers = []
    cols = len(ALL_METHODS) // 2 + 1
    classifier = Classificator()
    index = 1
    for method in ALL_METHODS:
        result[method] = classifier.classify(
            train, test, METHOD_TO_FUNCTION_MAP[method], use_database=use_database)

    voted_answers = []
    for i in range(len(test[0])):
        answers_to_image_1 = {}

        if SHOW:
            plt.subplot(3, cols, index)
            index += 1
            plt.imshow(cv2.cvtColor(test[0][i], cv2.COLOR_BGR2RGB)), plt.axis(
                "off"), plt.title("Selected image")

        for method in result:
            answer = result[method][i]
            if answer in answers_to_image_1:
                answers_to_image_1[answer] += 1
            else:
                answers_to_image_1[answer] = 1

            if SHOW:
                plt.subplot(3, cols, index)
                index += 1
                for train_image, true_answer in zip(train[0], train[1]):
                    if true_answer == answer:
                        plt.imshow(cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)), plt.axis(
                            "off"), plt.title(method)
                        break

        best_size = sorted(answers_to_image_1.items(),
                           key=lambda item: item[1], reverse=True)[0]
        voted_answers.append(best_size[0])

    return voted_answers
