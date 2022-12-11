import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import random

from cka import CKA
from simpleCNN import simpleCNN
from layerwiseCKA import layerwiseCKA

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
np.random.seed(0)
random.seed(0)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

batch_size = 256

dataset = CIFAR10(root='../data/',
                  train=False,
                  download=True,
                  transform=transform)

dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        worker_init_fn=seed_worker,
                        generator=g,)

folder_path = './trained_models/dyn_noniid/exp3'

cka = layerwiseCKA(dataloader = dataloader)
cka.load_client_models(load_path = folder_path)
cka.clientwise_cka()
cka.layerwise_cka()
cka.saving_plot(save_path = folder_path)
cka.print_average(save_path = folder_path)