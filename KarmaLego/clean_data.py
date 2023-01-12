import numpy as np
import pickle

def gradient (entity_list, original_data):
    gradient_list = np.ones((original_data.shape[0], original_data.shape[2], original_data.shape[1]-1))
    for k in range(original_data.shape[0]):
        for col in range(original_data.shape[2]):
            for row in range(original_data.shape[1]-1):
                if original_data[k][row + 1][col] == original_data[k][row][col]:
                    gradient_list[k][col][row] = 0
                if original_data[k][row + 1][col] < original_data[k][row][col]:
                    gradient_list[k][col][row] = -1
    # print(gradient_list[:2])
    for k in gradient_list:
        dict = {}
        for row in range(original_data.shape[2]):
            for col in range(original_data.shape[1]-1):
                if k[row][col] == -1:
                    index = row*9 + 1
                    if str(index) not in dict:
                        dict[str(index)] = []
                        tuple1 = (col*3+1, col*3+4)
                        dict[str(index)].append(tuple1)
                    else:
                        if dict[str(index)][-1][1] == col*3 + 1:
                            dict[str(index)][-1] = (dict[str(index)][-1][0], col*3+4)
                        else:
                            dict[str(index)].append((col*3+1, col*3+4))
                if k[row][col] == 0:
                    index = row*9 + 2
                    if str(index) not in dict:
                        dict[str(index)] = []
                        tuple1 = (col*3+1, col*3+4)
                        dict[str(index)].append(tuple1)
                    else:
                        if dict[str(index)][-1][1] == col*3+1:
                            dict[str(index)][-1] = (dict[str(index)][-1][0], col*3+4)
                        else:
                            dict[str(index)].append((col*3+1, col*3+4))
                if k[row][col] == 1:
                    index = row*9 + 3
                    if str(index) not in dict:
                        dict[str(index)] = []
                        tuple1 = (col*3+1, col*3+4)
                        dict[str(index)].append(tuple1)
                    else:
                        if dict[str(index)][-1][1] == col*3+1:
                            dict[str(index)][-1] = (dict[str(index)][-1][0], col*3+4)
                        else:
                            dict[str(index)].append((col*3+1, col*3+4))
        entity_list.append(dict)        
    return entity_list

def state(entity_list, original_data):
    state_list = np.ones((original_data.shape[0], original_data.shape[2], original_data.shape[1]-1))
    # normal_range = [60, 80, 36.2, 37.2, 110, 140, 70, 90, 7, 14, 7.34, 7.45, 38, 42, 96, 106,
    #                 70, 100, 4.5, 19.8, 2.4, 4.1, 0.2, 1.2, 11, 18, 4.5, 10, 200, 400, 150, 400]
    normal_range = [60,80,36.2,37.2,0.6,1.3,70,100]
    for k in range(original_data.shape[0]):
        round = 0
        for col in range(original_data.shape[2]):
            min = normal_range[round*2] 
            max = normal_range[round*2 + 1]
            round += 1
            for row in range(original_data.shape[1]-1):
                if original_data[k][row][col] < min:
                    state_list[k][col][row] = -1
                elif original_data[k][row][col] > max:
                    state_list[k][col][row] = 1
                else:
                    state_list[k][col][row] = 0
    # print(state_list[0])
    for k_index, k in enumerate(state_list):
        for row in range(original_data.shape[2]):
            for col in range(original_data.shape[1]-1):
                if k[row][col] == -1:
                    index = row*9 + 4
                    if str(index) not in entity_list[k_index]:
                        entity_list[k_index][str(index)] = []
                        tuple1 = (col*3+1, col*3+4)
                        entity_list[k_index][str(index)].append(tuple1)
                    else:
                        if  entity_list[k_index][str(index)][-1][1] == col*3+1:
                             entity_list[k_index][str(index)][-1] = ( entity_list[k_index][str(index)][-1][0], col*3+4)
                        else:
                             entity_list[k_index][str(index)].append((col*3+1, col*3+4))
                if k[row][col] == 0:
                    index = row*9 + 5
                    if str(index) not in entity_list[k_index]:
                        entity_list[k_index][str(index)] = []
                        tuple1 = (col*3+1, col*3+4)
                        entity_list[k_index][str(index)].append(tuple1)
                    else:
                       if  entity_list[k_index][str(index)][-1][1] == col*3+1:
                             entity_list[k_index][str(index)][-1] = ( entity_list[k_index][str(index)][-1][0], col*3+4)
                       else:
                             entity_list[k_index][str(index)].append((col*3+1, col*3+4))
                if k[row][col] == 1:
                    index = row*9 + 6
                    if str(index) not in entity_list[k_index]:
                        entity_list[k_index][str(index)] = []
                        tuple1 = (col*3+1, col*3+4)
                        entity_list[k_index][str(index)].append(tuple1)
                    else:
                        if  entity_list[k_index][str(index)][-1][1] == col*3+1:
                             entity_list[k_index][str(index)][-1] = ( entity_list[k_index][str(index)][-1][0], col*3+4)
                        else:
                             entity_list[k_index][str(index)].append((col*3+1, col*3+4)) 
    # print(entity_list[:5])
           
    return entity_list

def trend (entity_list, original_data):
    trend_list = np.zeros((len(original_data), original_data.shape[2], original_data.shape[1]-1))
    # normal_range = [10,0.5,10,10,3,0.05,2,5,10,1,0.5,0.5,2,1,50,50]
    normal_range = [10,0.5,2,10]
    for k in range(original_data.shape[0]):
        round = 0
        for col in range(original_data.shape[2]):
            delta = normal_range[round]
            round += 1
            for row in range(original_data.shape[1]-1):
               value_diff = original_data[k][row + 1][col] - original_data[k][row][col]
               if abs(value_diff) > delta:
                  if original_data[k][row + 1][col] > original_data[k][row][col]:
                    trend_list[k][col][row] = 1
                  else:
                    trend_list[k][col][row] = -1
    for k in range(trend_list.shape[0]):
        for col in range(original_data.shape[2]):
            for row in range(original_data.shape[1]-1):
                if trend_list[k][col][row] == 0:
                    if row == 0:
                        if trend_list[k][col][1] == 0:
                            row = 2
                            for i in range(row):
                                trend_list[k][col][i] = trend_list[k][col][row]
                        else:
                            trend_list[k][col][0] = trend_list[k][col][1]
                        # row += 1
                        # while row < 2:
                        #     if trend_list[k][col][row] == 0:
                        #         row += 1
                        #     else:
                        #         break
                        
                    else:
                        trend_list[k][col][row] = trend_list[k][col][row-1]
    # print(trend_list[0])
    for k_index, k in enumerate(trend_list):
        for row in range(original_data.shape[2]):
            for col in range(original_data.shape[1]-1):
                if k[row][col] == -1:
                    index = row*9 + 7
                    if str(index) not in entity_list[k_index]:
                        entity_list[k_index][str(index)] = []
                        tuple1 = (col*3 + 1, col*3 + 4)
                        entity_list[k_index][str(index)].append(tuple1)
                    else:
                        if  entity_list[k_index][str(index)][-1][1] == col*3 + 1:
                             entity_list[k_index][str(index)][-1] = ( entity_list[k_index][str(index)][-1][0], col*3 + 4)
                        else:
                             entity_list[k_index][str(index)].append((col*3 + 1, col*3 + 4))
                if k[row][col] == 0:
                    index = row*9 + 8
                    if str(index) not in entity_list[k_index]:
                        entity_list[k_index][str(index)] = []
                        tuple1 = (col*3 + 1, col*3 + 4)
                        entity_list[k_index][str(index)].append(tuple1)
                    else:
                        if  entity_list[k_index][str(index)][-1][1] == col*3 + 1:
                             entity_list[k_index][str(index)][-1] = ( entity_list[k_index][str(index)][-1][0], col*3 + 4)
                        else:
                             entity_list[k_index][str(index)].append((col*3 + 1, col*3 + 4))
                if k[row][col] == 1:
                    index = row*9 + 9
                    if str(index) not in entity_list[k_index]:
                        entity_list[k_index][str(index)] = []
                        tuple1 = (col*3 + 1, col*3 + 4)
                        entity_list[k_index][str(index)].append(tuple1)
                    else:
                        if  entity_list[k_index][str(index)][-1][1] == col*3 + 1:
                             entity_list[k_index][str(index)][-1] = ( entity_list[k_index][str(index)][-1][0], col*3 + 4)
                        else:
                             entity_list[k_index][str(index)].append((col*3 + 1, col*3 + 4))
           
    return entity_list             

def divide_label (data, label):
    sepsis_data = []
    non_sepsis_data = []
    for ele in range(len(label)):
        if label[ele] == 1:
            sepsis_data.append(data[ele])
        else:
            non_sepsis_data.append(data[ele])
    return np.array(sepsis_data),np.array(non_sepsis_data)


if __name__ == '__main__':
    data =np.load('/home/zhutou/project/karmalego/KarmaLego/data/train_origin.npy',encoding = "latin1")  
    label =np.load('/home/zhutou/project/karmalego/KarmaLego/data/train_logit.npy',encoding = "latin1")  
    # print(len(data))
    doc = open('all_temporal_origin.txt', 'w')  
    # print(data, file=doc)
    # print(len(data[0]))
    # data = data[:, :, [0,2,3,5,6,11,12,18,21,22,24,26,29,31,32,33]]
    data = data[:, :, [0,2,19,21]]
    print(data)
    data = data[:,[0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45],:]
    sepsis_data, non_sepsis_data = divide_label(data, label)
    # print(sepsis_data.shape[1])
    # print (len(sepsis_data))
    sepsis_entity_list = []
    non_spesis_entity_list = []

    sepsis_entity_list = gradient(sepsis_entity_list, sepsis_data)
    # print(sepsis_entity_list[:2])

    sepsis_entity_list = state(sepsis_entity_list, sepsis_data)
    sepsis_entity_list = trend(sepsis_entity_list, sepsis_data)

    non_spesis_entity_list = gradient(non_spesis_entity_list,non_sepsis_data)
    non_spesis_entity_list = state(non_spesis_entity_list, non_sepsis_data)
    non_spesis_entity_list = trend(non_spesis_entity_list, non_sepsis_data)
    non_spesis_entity_list = non_spesis_entity_list[:3000]
    # print (len(sepsis_entity_list))
    # entity_list = entity_list[:10]
    # print(sepsis_entity_list[0])
    f_sepsis = open("/home/zhutou/project/karmalego/KarmaLego/sepsis_entity_list.data","wb")
    f_non_sepsis = open("/home/zhutou/project/karmalego/KarmaLego/non_sepsis_entity_list.data","wb")
    pickle.dump(non_spesis_entity_list, f_non_sepsis)
    pickle.dump(sepsis_entity_list, f_sepsis)
    
    