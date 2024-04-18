import os
import torch
import re
import sys
import matplotlib.pyplot as plt
import numpy
file = '/home/qasim/Projects/TurboMedSAM/work_dir/DistillRepViT-ViTB_PreComputed/log_05042024-2058_DistillRepViT-ViTB_PreComputed.log'
window_size = 70
lines = []
with open(file, 'r') as reader:
    lines = reader.readlines()

total_losses = [line.split('total loss: ') for line in lines]
total_losses = [float(line[1].strip()) for line in total_losses if len(line) == 2]

def moving_average(
    data,
    window_size: int = None,
) -> numpy.ndarray:
    if type(data) == list:
        data = numpy.array(data)
    if window_size is None:
        window_size = len(data) // 10
    kernel = numpy.ones(window_size) / window_size
    smoothed_data = numpy.convolve(data, kernel, mode='valid')
    return smoothed_data
total_losses = moving_average(total_losses, window_size=window_size)

plt.plot(total_losses)
plt.title('total loss')
plt.show()
plt.savefig('total_loss.png')