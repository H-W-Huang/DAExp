import torch
import numpy as np
import random

import sys
import traceback

class Tee(object):
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self.stdout
        if exc_type is not None:
            self.file.write(traceback.format_exc())
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


def de_random(seed):
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

