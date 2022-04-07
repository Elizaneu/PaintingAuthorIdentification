# Parameters
from src.classification.features import sobel, Sato, fast


METHOD_SOBEL = "sobel"
METHOD_SATO = "Sato"
METHOD_FAST = "fast"

ALL_METHODS = (METHOD_SOBEL, METHOD_SATO, METHOD_FAST)

METHOD_TO_FUNCTION_MAP = {
    METHOD_SOBEL: sobel,
    METHOD_SATO: Sato,
    METHOD_FAST: fast,
}

CLASSES = {
    0: "Surikov",
    1: "Aivazovsky",
    2: "Shishkin",
    3: "Korovin",
    4: "Da Vinci"
}

IMAGES_PER_CLASS = 16
IMAGES_TRAINING = 8
