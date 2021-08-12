import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import time


def data_form_tran(data):
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    data_transformed = []
    data_transformed.append(x)
    data_transformed.append(y)
    data_transformed.append(z)
    return data_transformed

    
    
# FarthestPointSampling
def farthestPointSampling(points, samples_num):
    points = data_form_tran(points)
    
    select = []
    lens = len(points[0])
    rest = [num for num in range(0, lens)]
    max_dist = -1e10
    farthest_point = 1e10

    random.seed(1)
    ind = random.randint(0, lens)
    select.append(ind)
    rest.remove(ind)

    for i in range (lens):
        if i != ind:
            length = (points[0][ind] - points[0][i]) ** 2 + (points[1][ind] - points[1][i]) ** 2 + (points[2][ind] - points[2][i]) ** 2
            if length > max_dist:
                max_dist = length
                farthest_point = i
    select.append(farthest_point)
    rest.remove(farthest_point)

    while len(select) <  samples_num:
        min_length = []
        max_dist = -1e10

        for i in range(len(rest)):
            min_dist = 1e10

            for j in range(len(select)):
                length = (points[0][rest[i]] - points[0][select[j]]) ** 2 + (points[1][rest[i]] - points[1][select[j]]) ** 2 + (points[2][rest[i]] - points[2][select[j]]) ** 2
                if length < min_dist:
                    min_dist = length
                    #print(min_dist)

            min_length.append((rest[i], min_dist))

            if list(min_length[i])[1] > max_dist:
                max_dist = list(min_length[i])[1] 
                farthest_point = list(min_length[i])[0]

        select.append(farthest_point)
        rest.remove(farthest_point)
        
    return select 

if __name__ == "__main__":
    #displayPoint(data, "airplane")
    data = np.load('playground/temp.npy')
    start_cpu = time.time()
    selected_index = farthestPointSampling(data, 10)

    end_cpu = time.time()

    print('------ Cpu process time:' + str(end_cpu - start_cpu))
    #print(sample_data)