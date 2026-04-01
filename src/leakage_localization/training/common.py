from typing import Literal

LEAKAGE_MODEL = Literal[
    'bit',
    'id',
    'hw'
]
PHASE = Literal[
    'train',
    'val',
    'test'
]
PREPROCESSING = Literal[
    'standardize',
    'normalize'
]