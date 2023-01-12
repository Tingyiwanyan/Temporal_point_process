import numpy as np
import random

data =np.load('/home/zhutou/project/karmalego/KarmaLego/data/train_origin.npy',encoding = "latin1")  
# print(data[0][0].size)

for k in range(data.shape[0]):
    for row in range(data.shape[1]):
        for col in range(data.shape[2]):
            rand = random.uniform(0,1)
            if rand <= 0.2:
                data[k][row][col] = -1
print(data[:5])