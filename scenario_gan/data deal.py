import numpy as np
import torch




data = np.genfromtxt('D:\\scenario_gan\\DATASET.txt', delimiter=',', dtype=None).astype(float)
dataset = torch.tensor(data, dtype=torch.float32) / (data.max())
print(dataset.max())
dataset = dataset.reshape(365, 96, 3).mean(axis=2).cpu()
#dataset = dataset[:,]

print(dataset.shape) 
torch.save(dataset, "D:\\scenario_gan\\dataset.pth")