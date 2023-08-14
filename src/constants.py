import os

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

STAMP_SHAPE_MATRIX_PATH = os.path.join(ROOT_DIR, 'data', 'shape', 'doubledome_shape.npy')
ENCODING_PATH = os.path.join(ROOT_DIR, 'data', 'encoding')
LENGTHS_PATH = os.path.join(ROOT_DIR, 'Austausch_IPD_Optimierung', 'u_spring', 'U_spring.csv')

MATRIX_SHAPE = (460, 300)

INPUT_SHAPE = (60, )
OUTPUT_SHAPE = (460, 300, 1)