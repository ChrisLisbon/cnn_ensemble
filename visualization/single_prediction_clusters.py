import numpy as np
import torch
from matplotlib import pyplot as plt

from predict_ensemble import get_features_for_single_models, get_real_data, init_model
from root import root
from visualization.tricks import gradate, rotate
plt.rcParams['image.cmap'] = 'Blues'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prj_root = root()

start_point = '20110101'
sea_name = 'kara'
features = get_features_for_single_models(start_point, sea_name)
features = torch.Tensor(gradate(features)).to(device)

target, target_dates = get_real_data(start_point, sea_name)
target = gradate(target)

# ЗАМЕНИТЬ НА ОБУЧЕННЫЕ С 1990 ПО 2016
cnn1 = init_model(104, 52, (target.shape[1], target.shape[2]))
cnn1.load_state_dict(torch.load(f'{prj_root}/single_models/{sea_name}_104_52_clusters(1990-2010).pt'))
cnn1_prediction = cnn1(features).cpu().detach().numpy().astype('int')

coastline = rotate(np.load('../matrices/kara_land_mask.npy'))
x = np.arange(coastline.shape[1])
y = np.arange(coastline.shape[0])

for i in range(0, len(target_dates)):
    fig = plt.figure(constrained_layout=True, figsize=(11, 9))
    axs = fig.subplot_mosaic([['im1', 'im2']
                              ],
                             gridspec_kw={'width_ratios': [5, 5],
                                          'height_ratios': [9]})

    axs['im1'].set_title(f'{target_dates[i]} - real', c='r')
    axs['im1'].imshow(rotate(target[i]))
    axs['im1'].contour(x, y, coastline, [0], colors=['black'], linewidths=0.8)

    axs['im2'].set_title(f'Prediction', c='r')
    axs['im2'].imshow(rotate(cnn1_prediction[i]))
    axs['im2'].contour(x, y, coastline, [0], colors=['black'], linewidths=0.8)


    plt.show()
