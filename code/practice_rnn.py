import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

ac_data = pd.read_csv('acceleration_data.txt', sep=',', lineterminator=';', header=None, on_bad_lines='skip')
ac_data.columns = ['age', 'action', 'time', 'x', 'y', 'z']
print(ac_data.head(10))