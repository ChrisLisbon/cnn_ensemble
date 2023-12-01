import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

seas_codes = {0: 'Гренландское',
              1: 'Восточно-Сибирское',
              2: 'Чукотское',
              3: 'Бофорта',
              4: 'Лаптевых',
              5: 'Карское',
              6: 'Баренцево'
              }


def save_kara_sea_matrices():
    train_path = 'C:/Users/Julia/Documents/NSS_lab/autoencoder_ice_forecasting/full_arctic_mode/matrices/osisaf/train'
    test_path = 'C:/Users/Julia/Documents/NSS_lab/autoencoder_ice_forecasting/full_arctic_mode/matrices/osisaf/test'
    mask = np.load('../matrices/seas_mask_82degr.npy')
    plt.imshow(mask)
    plt.show()
    plt.imshow(mask[0:140, 130:250]) #140x120
    plt.show()
    for folder in [train_path, test_path]:
        for file in os.listdir(folder):
            date = file[-12:-4]
            matrix = np.load(f'{folder}/{file}')
            np.save(f'../matrices/kara_sea_osisaf/osi_kara_{date}.npy', matrix[0:140, 130:250])


def save_laptev_sea_matrices():
    train_path = 'C:/Users/Julia/Documents/NSS_lab/autoencoder_ice_forecasting/full_arctic_mode/matrices/osisaf/train'
    test_path = 'C:/Users/Julia/Documents/NSS_lab/autoencoder_ice_forecasting/full_arctic_mode/matrices/osisaf/test'
    mask = np.load('../matrices/seas_mask_82degr.npy')
    plt.imshow(mask)
    plt.show()
    plt.imshow(mask[50:160, 210:340]) #110x130
    plt.show()
    for folder in [train_path, test_path]:
        for file in os.listdir(folder):
            date = file[-12:-4]
            matrix = np.load(f'{folder}/{file}')[50:160, 210:340]
            np.save(f'../matrices/laptev_sea_osisaf/osi_laptev_{date}.npy', matrix)


def prepare_mask():
    dates = pd.date_range('19900101', '20091231')
    dates = [date.strftime('%Y%m%d') for date in dates]
    matrix_sum = np.zeros((140, 120))
    for date in dates:
        matrix = np.load(f'../matrices/kara_sea_osisaf/osi_kara_{date}.npy')
        matrix_sum = matrix_sum+matrix
    matrix_sum[matrix_sum != 0] = 1
    '''matrix_sum[68:, :40] = 1
    matrix_sum[80:, :54] = 1
    matrix_sum[60:, :5] = 1'''
    plt.imshow(matrix_sum)
    plt.show()
    np.save('../matrices/kara_land_mask.npy', matrix_sum)
