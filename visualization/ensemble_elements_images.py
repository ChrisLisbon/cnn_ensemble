import numpy as np
from matplotlib import pyplot as plt

from predict_ensemble import get_ensemble_prediction
from visualization.tricks import rotate
plt.rcParams['image.cmap'] = 'Blues'


start_point = '20160101'
ensemble_forecast, cnn1_prediction, cnn2_prediction, baseline_prediction, target, target_dates = get_ensemble_prediction(start_point)


coastline = rotate(np.load('../matrices/kara_land_mask.npy'))
x = np.arange(coastline.shape[1])
y = np.arange(coastline.shape[0])

for i in range(len(target_dates)):
    fig = plt.figure(constrained_layout=True, figsize=(11, 7))
    axs = fig.subplot_mosaic([['im1', 'im2', 'im3'],
                              ['im1', 'im4', 'im5']
                              ],
                             gridspec_kw={'width_ratios': [5, 5, 5],
                                          'height_ratios': [7, 7]})

    axs['im1'].set_title(f'{target_dates[i]} - real', c='r')
    axs['im1'].imshow(rotate(target[i]), vmax=1, vmin=0)
    axs['im1'].contour(x, y, coastline, [0], colors=['black'], linewidths=0.8)
    axs['im1'].contour(x, y, rotate(target[i]), [0.8], colors=['lime'])

    axs['im2'].set_title(f'Meanyears', c='r')
    axs['im2'].imshow(rotate(baseline_prediction[i]), vmax=1, vmin=0)
    axs['im2'].contour(x, y, coastline, [0], colors=['black'], linewidths=0.8)
    axs['im2'].contour(x, y, rotate(target[i]), [0.8], colors=['lime'])
    axs['im2'].contour(x, y, rotate(baseline_prediction[i]), [0.8], colors=['r'])

    axs['im3'].set_title(f'L1 CNN', c='r')
    axs['im3'].imshow(rotate(cnn1_prediction[i]), vmax=1, vmin=0)
    axs['im3'].contour(x, y, coastline, [0], colors=['black'], linewidths=0.8)
    axs['im3'].contour(x, y, rotate(target[i]), [0.8], colors=['lime'])
    axs['im3'].contour(x, y, rotate(cnn1_prediction[i]), [0.8], colors=['r'])

    axs['im4'].set_title(f'SSIM CNN', c='r')
    axs['im4'].imshow(rotate(cnn2_prediction[i]), vmax=1, vmin=0)
    axs['im4'].contour(x, y, coastline, [0], colors=['black'], linewidths=0.8)
    axs['im4'].contour(x, y, rotate(target[i]), [0.8], colors=['lime'])
    axs['im4'].contour(x, y, rotate(cnn2_prediction[i]), [0.8], colors=['r'])

    axs['im5'].set_title(f'Ensemble', c='r')
    axs['im5'].imshow(rotate(ensemble_forecast[i]), vmax=1, vmin=0)
    axs['im5'].contour(x, y, coastline, [0], colors=['black'], linewidths=0.8)
    axs['im5'].contour(x, y, rotate(target[i]), [0.8], colors=['lime'])
    axs['im5'].contour(x, y, rotate(ensemble_forecast[i]), [0.8], colors=['r'])

    plt.show()
