from tkinter import filedialog

from matplotlib import pyplot as plt
from src.classification.voting import voting
from src.classification.constants import *
from src.utils.data_preprocessor import DataPreprocessor
from src.ui.window import Window
from src.ui.ui import UI
from src.classification.classificator import Classificator
from src.classification.features import *
from src.screens.warning import Warning
from cv2 import *

# Images uploading
T_IMAGES_UPLOADING = "T_IMAGES_UPLOADING"
L_IMAGES_UPLOADED_HEAD = "LABEL_IMAGES_UPLOADED_HEAD"
L_IMAGES_UPLOADED_VALUE = "LABEL_IMAGES_UPLOADED_VALUE"
B_UPLOAD_IMAGES = "B_UPLOAD_IMAGES"

# Identification
T_IDENTIFICATION = "T_IDENTIFICATION"
B_IDENTIFY_AUTHOR = "B_IDENTIFY_AUTHOR"
B_IDENTIFY = "B_IDENTIFY"
B_PRINT_METHODS = "B_PRINT_METHODS"
B_TRAINING_TEST = "B_TRAINING_TEST"


class MainScreen(Window):
    def __init__(self) -> None:
        super().__init__("Portrait Author Identification")

        self.__selected_image = []

        self.__init_components()

        # Identification
        self.__is_data_loaded = False
        self.__data = []

    # UI
    def __init_components(self):
        # Images uploading
        self.include_component(
            T_IMAGES_UPLOADING,
            UI.get_title("Images uploading", self._get_next_row())
        )
        self.include_component(
            L_IMAGES_UPLOADED_HEAD,
            UI.get_label("Images status:", self._get_next_row()),
        )
        self.include_component(
            L_IMAGES_UPLOADED_VALUE,
            UI.get_label("not loaded", self._get_row(), 1),
        )
        self.include_component(
            B_UPLOAD_IMAGES,
            UI.get_button(
                "Load images", self.__handle_images_load, self._get_next_row()
            ),
        )

        # Identification
        self.include_component(
            T_IDENTIFICATION,
            UI.get_title("Author identification", self._get_next_row()),
        )
        self.include_component(
            B_IDENTIFY,
            UI.get_button("Test Identify", self.__handle_test_classify,
                          self._get_next_row())
        )
        self.include_component(
            B_IDENTIFY_AUTHOR,
            UI.get_button("Identify Author", self.__handle_classify,
                          self._get_row(), column=1)
        )
        self.include_component(
            B_PRINT_METHODS,
            UI.get_button("Print Image", self.__handle_print_methods_to_image,
                          self._get_row(), column=2)
        )
        self.include_component(
            B_TRAINING_TEST,
            UI.get_button("Training test identify",
                          self.__handle_to_training_test, self._get_row(), column=4)
        )
        self._get_next_row()
        self._get_next_row()

    def __handle_images_load(self) -> None:
        self.__data = DataPreprocessor.load_images('./paintings/s')
        self.__is_data_loaded = True

        label = self.get_component(L_IMAGES_UPLOADED_VALUE)
        label.configure(text="loaded")

    def __handle_test_classify(self) -> None:
        if not self.__is_data_loaded:
            modal = Warning("You need to load images before identification!")
            modal.open()
            return

        x_train, x_test, y_train, y_test = DataPreprocessor.split_data_randomly(
            self.__data, IMAGES_PER_CLASS, IMAGES_TRAINING, True)
        train = [x_train, y_train]
        test = [x_test, y_test]
        classificator = Classificator()
        methods = []
        for method in ALL_METHODS:
            methods.append(METHOD_TO_FUNCTION_MAP[method])
        featured_train = {method: classificator.create_feature(
            x_train, method) for method in methods}

        for test_img in test[0]:
            answer = voting(train, [[test_img], [0]],
                            SHOW=True, use_database=featured_train)
            plt.subplot(3, 3, 8, title="Author"), plt.axis("off")
            plt.text(0.3,
                     0.5,
                     CLASSES[answer[0]],
                     transform=plt.gca().transAxes, fontdict={'size': 12})
            plt.show()

    def __handle_classify(self) -> None:
        if not self.__is_data_loaded:
            modal = Warning("You need to load images before identification!")
            modal.open()
            return

        x_train, _, y_train, _ = DataPreprocessor.split_data_randomly(
            self.__data, IMAGES_PER_CLASS, IMAGES_TRAINING, True)
        train = [x_train, y_train]
        classificator = Classificator()
        methods = []
        for method in ALL_METHODS:
            methods.append(METHOD_TO_FUNCTION_MAP[method])
        featured_train = {method: classificator.create_feature(
            x_train, method) for method in methods}

        image_path = filedialog.askopenfilename(initialdir="./",
                                                title="Select a File",
                                                filetypes=[("Image files", "*.jpg *.png *.pgm")])
        self.update()

        image = cv2.imread(image_path)
        self.__selected_image = image

        answer = voting(train, [[self.__selected_image], [0]],
                        SHOW=True, use_database=featured_train)
        plt.subplot(3, 3, 8, title="Author"), plt.axis("off")
        plt.text(0.3,
                 0.5,
                 CLASSES[answer[0]],
                 transform=plt.gca().transAxes, fontdict={'size': 12})
        plt.show()

    def __handle_print_methods_to_image(self) -> None:
        if not len(self.__selected_image):
            modal = Warning(
                "You need to select an image before printing methods to image!")
            modal.open()
            return
        plt.rcParams["figure.figsize"] = (8, 8)
        index = 1
        cols = len(ALL_METHODS) // 2 + 1
        plt.subplot(2, cols, index, title="original")
        plt.imshow(cv2.cvtColor(self.__selected_image,
                   cv2.COLOR_BGR2RGB)), plt.axis("off")

        for method in ALL_METHODS:
            index += 1
            plt.subplot(2, cols, index, title=method)

            to_draw = METHOD_TO_FUNCTION_MAP[method](self.__selected_image)[1]
            if len(to_draw.shape) == 2:
                plt.imshow(to_draw, cmap="gray"), plt.axis("off")
            else:
                plt.imshow(cv2.cvtColor(to_draw, cv2.COLOR_BGR2RGB)
                           ), plt.axis("off")
        plt.show()

    def __handle_to_training_test(self) -> None:
        if not self.__is_data_loaded:
            modal = Warning("You need to load images before identification!")
            modal.open()
            return

        x_train, _, y_train, _ = DataPreprocessor.split_data_randomly(
            self.__data, IMAGES_PER_CLASS, IMAGES_TRAINING, True)
        train = [x_train, y_train]
        classificator = Classificator()
        methods = []
        for method in ALL_METHODS:
            methods.append(METHOD_TO_FUNCTION_MAP[method])
        featured_train = {method: classificator.create_feature(
            x_train, method) for method in methods}

        for train_img in train[0]:
            answer = voting(train, [[train_img], [0]],
                            SHOW=True, use_database=featured_train)
            plt.subplot(3, 3, 8, title="Author"), plt.axis("off")
            plt.text(0.3,
                     0.5,
                     CLASSES[answer[0]],
                     transform=plt.gca().transAxes, fontdict={'size': 12})
            plt.show()
