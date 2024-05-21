# /data/xfni/code/anaconda/bin/python
# -*- coding: utf-8 -*-
# @Time         : 2023/3/20 15:37
# @Author       : patrick
# @File         : take_up_gpu.py
# @Description  : Fuck those who try to steal the GPUs from me!
import argparse

import torch
import time

parser = argparse.ArgumentParser()
parser.add_argument('--g0', action='store_true')
parser.add_argument('--g1', action='store_true')
parser.add_argument('--g2', action='store_true')
parser.add_argument('--g3', action='store_true')
parser.add_argument('--g4', action='store_true')
parser.add_argument('--g5', action='store_true')
parser.add_argument('--g6', action='store_true')
parser.add_argument('--g7', action='store_true')
parser.add_argument('--sleep', type=float)
parser.add_argument('--num', type=int)
parser.add_argument('--burn', action='store_true')
args = parser.parse_args()

gpus = []
if args.g0:
    gpus.append('0')
if args.g1:
    gpus.append('1')
if args.g2:
    gpus.append('2')
if args.g3:
    gpus.append('3')
if args.g4:
    gpus.append('4')
if args.g5:
    gpus.append('5')
if args.g6:
    gpus.append('6')
if args.g7:
    gpus.append('7')
dogs = []
for gpu in gpus:
    if args.burn:
        dogs.append([torch.randn([4200,4200]).to(f"cuda:{gpu}") for _ in range(args.num)])
    else:
        dogs.append([torch.randn([1400,1400]).to(f"cuda:{gpu}") for _ in range(args.num)])
    print(f"{gpu}号卡【DOGE】")

count = 0
all_gpu = '、'.join(gpus)
print(f"忠心为您守护{all_gpu}号显卡！")

while 1:
    for dog in dogs:
        for i in range(1200):
            dog[0] = dog[1] * dog[2]
        time.sleep(args.sleep)
