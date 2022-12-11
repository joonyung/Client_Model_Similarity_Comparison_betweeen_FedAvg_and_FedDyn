import torch
from torchvision.models import resnet18, resnet34, resnet50, wide_resnet50_2
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import random

from cka import CKA
from vggnet import VGG, MyBatchNorm

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
np.random.seed(0)
random.seed(0)

model1 = resnet18(pretrained=True)
model2 = resnet34(pretrained=True)

# batch_norm = MyBatchNorm
# model1 = VGG(n_blocks = 4, norm_layer = batch_norm, num_classes = 10)
# model2 = VGG(n_blocks = 4, norm_layer = batch_norm, num_classes = 10)

# model1.load_state_dict(torch.load('./testModel0.pt'))
# model2.load_state_dict(torch.load('./testModel1.pt'))

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
        device='cuda')

cka.compare(dataloader)

#cka.plot_results(save_path="../assets/TEST1.png")
cka.printing_named_modules()

#===============================================================
# model1 = resnet50(pretrained=True)
# model2 = wide_resnet50_2(pretrained=True)


# cka = CKA(model1, model2,
#         model1_name="ResNet50", model2_name="WideResNet50",
#         device='cuda')

# cka.compare(dataloader)

# cka.plot_results(save_path="../assets/resnet-resnet_compare.png")