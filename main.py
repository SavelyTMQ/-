import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
import math

import os
import random
import time
from sklearn.model_selection import train_test_split

def main():
    #excel_data = pd.read_excel('MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.xlsx')
    #excel_data = pd.read_excel('MachineLearningCVE/pr.xlsx')


    #считать все файлы из папки
    #directory = '/Users/User/Desktop/ДИПЛОМ/MachineLearningCVE'
    #files = os.listdir(directory)
    #print(files)

    S = []

    with open('MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', newline='') as File:
        reader = csv.reader(File)
        i = 0
        for row in reader:
            if i > 0:
                S.append(row)
            """print(type(row))
            print(len(row))
            if i == 2:
                break"""
            i += 1
    """print(len(S))
    k = round(len(S) * 0.1)
    
    Array_rand = np.random.randint(0, len(S), k)"""

    S = np.array(S)

    ind = np.arange(len(S))
    np.random.shuffle(ind)

    ind_train = ind[:round(0.1*len(ind))]
    ind_test = ind[round(0.1*len(ind)):]

    x_train = S[ind_train]
    x_test = S[ind_test]

    print(len(x_train), len(x_test))

# нормализуем данные [0, 1]
def data_normalization(data):
    scaler = preprocessing.MinMaxScaler()

    names = data.columns
    d = scaler.fit_transform(data)

    scaled_df = pd.DataFrame(d, columns=names)
    return scaled_df

# приводим данные к одному типу, чтобы их потом нормализовать
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

# аффиность антиген-антитело
def affinity(Ab, Ag):
    Ab = (Ab - Ag) ** 2
    D = math.sqrt(np.sum(Ab))
    return D

def aiNET(C_initial, a, ct, nt, B, mut):
    c = len(C_initial)
    Aff = np.array([])

    # вычисляем аффиность Ab-Ag
    for j in range(c):
        Aff = np.append(Aff, affinity(a, C_initial[j]))

    C_help = C_initial
    len_C = len(C_help)
    Aff = np.sort(Aff)
    C_help = C_help[Aff.argsort()]
    C_new = np.zeros((1, 25))
    Al = Aff <= ct
    Al = Al[Al == 1]

    # по 1 варинату обучение
    B = (B * len_C)
    new_pop = np.array([int(B / (i + 1)) for i in range(len(Al))])


    # по 2 варианту обучение
    #B = (B * len(Al)) / len_C
    #new_pop = np.array([int((B * len_C) / (i + 1)) for i in range(len(Al))])


    new_pop = new_pop[new_pop != 0]

    for i in range(len(new_pop)):
        D = np.random.uniform(low=mut[0], high=mut[1], size=(new_pop[i], 25))
        R = np.repeat(np.array([C_help[i]]), new_pop[i], axis=0)
        C_new = np.vstack((C_new, R + D))

    C_new = C_new[1:]

    # вычисляем аффиность клонов и антигена, для дальнейшего отсеивания
    c = len(C_new)
    Aff = np.array([])

    # вычисляем аффиность Ab-Ag
    for j in range(c):
        Aff = np.append(Aff, affinity(a, C_new[j]))

    C_help = C_new

    # добавлем в новую популяцию только клонов с аффиностью меньше пороговой
    C_new = C_help[Aff < nt]

    # добавляем клонов в старую популяци
    if len(C_new) != 0:
        C_initial = np.vstack((C_initial, C_new))

    # добавляем также антиген
    C_initial = np.vstack((C_initial, a))

    return C_initial

# Циклы обучения
def main_sup(np_attack, np_no_attack, N, ct, nt, TSC, B, mut, M, aff):

    # перемешиваем обучающую выборку
    ind = np.arange(len(np_attack))
    np.random.shuffle(ind)
    np_attack = np_attack[ind]

    ind = np.arange(len(np_no_attack))
    np.random.shuffle(ind)
    np_no_attack = np_no_attack[ind]

    no_train = np_no_attack[:int(len(np_no_attack) * 0.8)]

    yes_test = np_attack
    no_test = np_no_attack[int(len(np_no_attack) * 0.8):]

    # создаем начальные популяции
    C_no = no_train[:N]
    no_train = np.delete(no_train, range(N), axis=0)

    i = 0
    while len(C_no) < 100000 and i < len(no_train):
        C_no = aiNET(C_no, no_train[i], ct, nt, B, mut)
        print(f"Шаг {i + 1}")
        print(f"Норм попул {len(C_no)}")
        i += 1
    TSC = i

    # нахоим центр антител
    c_mid = np.sum(C_no, axis=0) / len(C_no)


    if len(no_test) >= M // 2:
        a = np.vstack((yes_test[:M // 2],  no_test[:M // 2]))
    elif len(np_no_attack) >= M // 2:
        a = np.vstack((yes_test[:M // 2], np_no_attack[:M // 2]))
    else:
        a = np.vstack((yes_test[:M // 2], np_no_attack))


    a = np.array([affinity(i, c_mid) for i in a])
    A_no = np.zeros(len(a))
    f = list(a[:M // 2] > aff)
    f = f + list(a[M // 2:] <= aff)
    A_no[f] = 1

    return C_no, np.sum(A_no[:M // 2]) / 1000 + ((np.sum(A_no[M // 2:]) / len(A_no[M // 2:]))) / 2, len(C_no), c_mid, np.sum(A_no[:int(M // 2)]), np.sum(A_no[int(M // 2):]), TSC



if __name__ == '__main__':
    #main()
    #main_sup()
    """A = np.array([[1, 2, 3], [3, 4, 5]], dtype=float)
    B = np.array([i + random.uniform(-0.1, 0.1) for i in A[0]])
    print(B)
    A = np.vstack((A, B))
    #A = np.delete(A, 1, 0)
    print(A)"""

    A = np.array([[1, 2, 3],
                  [3, 4, 5]])

    B = np.array([[1, 1, 1],
                  [30, 40, 50]])
    print(A[1:])
