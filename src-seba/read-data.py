import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import time


def read_df_as_clean_arrays(dfs, row_range = 300, col_range=(3,12)):
    data = []
    for d in dfs:
        data.append(d.values[:row_range,col_range[0]:col_range[1]])
    return data

def scalings(listas_arrays):

    aux = np.zeros(shape=(1,1))
    for d in listas_arrays:
        aux = np.vstack([aux, d.reshape(-1,1)])
    aux = aux[1:,:]
    print(aux.shape)
    scaler = StandardScaler()
    scaler.fit(aux)

    return scaler
    

if __name__  == '__main__':
    folder = os.path.join('..', 'Date')
    dataframes1 = []
    for f in os.listdir(folder):
        dataframes1.append(pd.read_excel(os.path.join(folder, f)))

    dataframes2 = []
    folder = os.path.join('..', 'Date2')
    for f in os.listdir(folder):
        print(f)
        fff = pd.read_excel(os.path.join(folder, f))
        dataframes2.append(fff)

    data_1  = read_df_as_clean_arrays(dataframes1, row_range=300, col_range=(3,12))

    data_2 = read_df_as_clean_arrays(dataframes2, row_range=99,col_range=(2,5))

    data_total = data_1[:]
    data_total.extend(data_2)
    print(len(data_1), len(data_2), len(data_total))

    scaler = scalings(data_total)

