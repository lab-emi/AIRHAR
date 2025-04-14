
from modules.data_collection import load_Radardataset
from utils.micro_Doppler import aug_rangefft, batch_loadnp
import numpy as np
import matplotlib.pyplot as plt
train_loader = load_Radardataset('/home/yizhuo/PycharmProjects/AIHAR/data', dataset_name='CI4R',subset='train', batch_size=1, aug_input=True,
                                     num_gpus=1)
val_loader = load_Radardataset('/home/yizhuo/PycharmProjects/AIHAR/data', dataset_name='CI4R', subset='val', batch_size=1, aug_input=False, num_gpus=1)
test_loader = load_Radardataset('/home/yizhuo/PycharmProjects/AIHAR/data', dataset_name='CI4R',subset='test', batch_size=1, aug_input=False,
                                    num_gpus=1)

for datapath, label in train_loader:
    input = batch_loadnp(datapath)
    _, rp, _ = aug_rangefft(input[0])
    range_noise = np.mean(rp, axis=-1)
    plt.plot(np.abs(range_noise[0]))
    plt.show()