import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_ice_extent_kara_sea():
    mask = np.load('../matrices/seas_mask_82degr.npy')[0:140, 130:250]
    key = 5
    #mask[mask != key] = np.nan
    plt.imshow(mask)
    plt.show()
    dates = []
    areas = []
    for file in os.listdir('../matrices/kara_sea_ensemble_prediction'):
        array = np.load(f'../matrices/kara_sea_ensemble_prediction/{file}')
        dates.append(file[-12:-4])
        array[mask != key] = np.nan
        ice_pixels = array[array >= 0.8]
        ice_area = 14 * 14 * ice_pixels.shape[0]
        areas.append(ice_area)
    df = pd.DataFrame()
    df['dates'] = pd.to_datetime(dates, format='%Y%m%d')
    df['kara_ensemble_ice_area'] = areas
    return df


def get_real_ice_extent_kara_sea():
    mask = np.load('../matrices/seas_mask_82degr.npy')[0:140, 130:250]
    key = 5
    plt.imshow(mask)
    plt.show()
    dates = []
    areas = []
    for file in os.listdir('../matrices/kara_sea_osisaf'):
        if int(file[-12:-8]) > 2015:
            array = np.load(f'../matrices/kara_sea_osisaf/{file}')
            dates.append(file[-12:-4])
            array[mask != key] = np.nan
            ice_pixels = array[array >= 0.8]
            ice_area = 14 * 14 * ice_pixels.shape[0]
            areas.append(ice_area)
    df = pd.DataFrame()
    df['dates'] = pd.to_datetime(dates, format='%Y%m%d')
    df['kara_real_ice_area'] = areas
    return df

def get_baseline_ice_extent_kara_sea():
    mask = np.load('../matrices/seas_mask_82degr.npy')[0:140, 130:250]
    key = 5
    plt.imshow(mask)
    plt.show()
    dates = []
    areas = []
    for file in os.listdir('../matrices/kara_sea_meanyears_prediction'):
        if int(file[-12:-8]) > 2015:
            array = np.load(f'../matrices/kara_sea_meanyears_prediction/{file}')
            dates.append(file[-12:-4])
            array[mask != key] = np.nan
            ice_pixels = array[array >= 0.8]
            ice_area = 14 * 14 * ice_pixels.shape[0]
            areas.append(ice_area)
    df = pd.DataFrame()
    df['dates'] = pd.to_datetime(dates, format='%Y%m%d')
    df['kara_baseline_ice_area'] = areas
    return df


seas5_df = pd.read_csv('C:/Users/Julia/Documents/NSS_lab/autoencoder_ice_forecasting/seas_based_statement/ice_extent_per_sea/ices_areas_ts_seas5(2020-2023).csv')
seas5_df['dates'] = pd.to_datetime(seas5_df['dates'])
ens_df = get_ice_extent_kara_sea()
#ens_df.to_csv('ice_area_ts_ensemble.csv', index=False)
real_df = get_real_ice_extent_kara_sea()
baseline_df = get_baseline_ice_extent_kara_sea()


plt.plot(ens_df['dates'], ens_df['kara_ensemble_ice_area'], label='ensemble')
plt.plot(real_df['dates'], real_df['kara_real_ice_area'], label='real')
plt.plot(baseline_df['dates'], baseline_df['kara_baseline_ice_area'], label='baseline')
plt.plot(seas5_df['dates'], seas5_df['Карское'], label='seas5')
plt.title('Ice area')
plt.legend()
plt.show()