import os
import pickle
import time
from datetime import datetime

from matplotlib import pyplot as plt
from pytorch_msssim import ssim
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from builder.EncoderForecasterBase import EncoderForecasterBase
from builder.TensorBuilder import multioutput_tensor, multioutput_numpy
from visualization.tricks import gradate

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Calculating on device: {device}')
batch_size = 10
epochs = 500
learning_rate = 1e-3

data_freq = 7
sea_name = 'kara'

DIMS = None
x_virg = []
for file in os.listdir(f'matrices/{sea_name}_sea_osisaf'):
    date = datetime.strptime(file, f'osi_{sea_name}_%Y%m%d.npy')
    if date.year < 2010:
        array = np.load(f'matrices/{sea_name}_sea_osisaf/{file}')
        DIMS = array.shape
        x_virg.append(array)
    else:
        break

x_virg = np.array(x_virg)[::7]
x_virg = gradate(x_virg)

pre_history_size = 104
forecast_size = 52

x, y = multioutput_numpy(pre_history_size, forecast_size, x_virg)
tensor_x = torch.Tensor(x)
tensor_y = torch.Tensor(y)
dataset = TensorDataset(tensor_x, tensor_y)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
print('Loader created')

encoder = EncoderForecasterBase()
encoder.init_encoder(input_size=[DIMS[0], DIMS[1]],
                     n_layers=5,
                     in_channels=pre_history_size,
                     out_channels=forecast_size)
encoder.to(device)
print(encoder)

optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

losses = []
start = time.time()
for epoch in range(epochs):
    loss = 0
    for train_features, test_features in dataloader:
        train_features = train_features.to(device)
        test_features = test_features.to(device)
        optimizer.zero_grad()
        outputs = encoder(train_features)
        train_loss = criterion(outputs, test_features)
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()

    loss = loss / len(dataloader)
    losses.append(loss)
    print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))

end = time.time() - start
print(f'Runtime seconds: {end}')
torch.save(encoder.state_dict(), f"single_models/{sea_name}_104_52_clusters(1990-2010).pt")
plt.plot(np.arange(len(losses)), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
