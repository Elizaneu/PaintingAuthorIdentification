import random as rnd
import cv2


class DataPreprocessor:
    def load_images(path, ext=".jpg"):
        dt_images = []
        dt_targets = []

        for i in range(1, 6):
            for j in range(1, 17):
                image = cv2.imread(f"{path}{i}/{j}{ext}")
                if image is not None:
                    dt_images.append(image)
                    dt_targets.append(i - 1)
        return [dt_images, dt_targets]

    def split_data(data, images_per_class=16, images_per_class_in_train=8):
        images_all = len(data[0])

        x_train, x_test, y_train, y_test = [], [], [], []

        for i in range(0, images_all, images_per_class):
            x_train.extend(data[0][i: i + images_per_class_in_train])
            y_train.extend(data[1][i: i + images_per_class_in_train])

            x_test.extend(
                data[0][i + images_per_class_in_train: i + images_per_class])
            y_test.extend(
                data[1][i + images_per_class_in_train: i + images_per_class])

        return x_train, x_test, y_train, y_test

    def split_data_randomly(data, images_per_class=16, images_per_class_in_train=8, SHOW=False):
        amount_of_images = len(data[0])
        x_train, x_test, y_train, y_test = [], [], [], []

        for i in range(0, amount_of_images, images_per_class):
            indexes = list(range(i, i + images_per_class))
            train_indexes = rnd.sample(indexes, images_per_class_in_train)
            if SHOW:
                print(
                    f"train_indexes: {[(ind + 1) % 17 for ind in train_indexes]}")
            x_train.extend([data[0][index] for index in train_indexes])
            y_train.extend([data[1][index] for index in train_indexes])

            test_indexes = set(indexes) - set(train_indexes)
            if SHOW:
                print(
                    f"test_indexes: {[(ind + 1) % 17 for ind in test_indexes]}")
            x_test.extend([data[0][index] for index in test_indexes])
            y_test.extend([data[1][index] for index in test_indexes])

        return x_train, x_test, y_train, y_test


DataPreprocessor.load_images = staticmethod(
    DataPreprocessor.load_images)
DataPreprocessor.split_data = staticmethod(DataPreprocessor.split_data)
DataPreprocessor.split_data_randomly = staticmethod(
    DataPreprocessor.split_data_randomly)
