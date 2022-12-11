import torch
#from torchvision.models import resnet18, resnet34, resnet50, wide_resnet50_2
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import random

from cka import CKA
from simpleCNN import simpleCNN

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
np.random.seed(0)
random.seed(0)

model1 = simpleCNN()
model2 = simpleCNN()

model1.load_state_dict(torch.load('./client0.pt'))
model2.load_state_dict(torch.load('./client1.pt'))

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

cka = CKA(model1, model2,
        model1_name="VGG1", model2_name="VGG2",
        model1_layers=['conv1', 'conv2'], model2_layers=['conv1', 'conv2'],
        device='cuda')

cka.printing_named_modules()

cka.compare(dataloader)

#cka.plot_results(save_path="../assets/TEST3.png")

#print(cka.export())