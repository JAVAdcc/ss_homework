import torch
import numpy as np

def ss_conv1d(inp, kernel) -> np.array:
    # 步长固定为1 padding需要结合kernel的size决定
    # 最终输出长度为 inp_length + kernel_length - 1
    stride = 1
    padding = kernel.shape[0] - 1
    out_length = inp.shape[0] + kernel.shape[0] - 1

    # 反转卷积核
    kernel = np.flip(kernel)
    # 给inp补0
    inp = np.pad(inp, padding, 'constant')

    # 进行卷积操作
    out = np.zeros(out_length)
    for i in range(out_length):
        out[i] = np.dot(kernel, inp[i:i+kernel.shape[0]])

    return out

def ss_conv2d(inp, kernel) -> np.array:
    # 步长固定为1 padding需要结合kernel的size决定
    # 最终输出尺寸为 (inp_w + kernel_w - 1, inp_h + kernel_h - 1)
    stride = 1
    padding_h = kernel.shape[0] - 1
    padding_w = kernel.shape[1] - 1
    out_h = inp.shape[0] + kernel.shape[0] - 1
    out_w = inp.shape[1] + kernel.shape[1] - 1

    # 反转卷积核
    kernel = np.flip(kernel)
    # 给inp补0
    inp = np.pad(inp, ((padding_h, padding_h), (padding_w, padding_w)), mode='constant')

    # 进行卷积操作
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = np.dot(kernel.flatten(), inp[i:i+kernel.shape[0], j:j+kernel.shape[1]].flatten())

    return out