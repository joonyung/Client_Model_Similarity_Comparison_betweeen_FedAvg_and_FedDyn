import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils import add_colorbar
from simpleCNN import simpleCNN
from cka import CKA


class layerwiseCKA:
    def __init__(self, dataloader: DataLoader, num_of_clients = 10):
        self.dataloader = dataloader
        self.num_of_clients = num_of_clients

        self.model_layer_label = ['conv1', 'conv2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7']
        self.num_of_layers = len(self.model_layer_label)
        self.client_models = {}     #{(int) model_id : model}
        self.clientwise_cka_matrix = {}     #{(list) client_pair(i,j) : (tensor) CKA_matrix}
        self.layerwise_cka_matrix = {}      #{(int) layer_id : (tensor) CKA_matrix}
        for i in range(self.num_of_layers):
            self.layerwise_cka_matrix[i] = torch.zeros(self.num_of_clients, self.num_of_clients)
        self.average_cka = {}    #{(int) layer_id : (float) average cka}
        

    def load_client_models(self, load_path: str = None):
        for i in range(self.num_of_clients):
            client_model = simpleCNN()
            file_name = load_path + '/client{}.pt'.format(i)
            client_model.load_state_dict(torch.load(file_name))
            self.client_models[i] = client_model

    def clientwise_cka(self):
        for i in range(self.num_of_clients):
            for j in range(i + 1):
                cka = CKA(self.client_models[i], self.client_models[j], 
                        model1_name = "Client{}".format(i), model2_name = "Client{}".format(j), 
                        model1_layers = self.model_layer_label, model2_layers = self.model_layer_label,
                        device = 'cuda')

                cka.compare(self.dataloader)
                
                client_pair = (i, j)
                self.clientwise_cka_matrix[client_pair] = cka.CKA_matrix()

    def layerwise_cka(self):
        for client_pair, cka_matrix in self.clientwise_cka_matrix.items():
            for layer in range(self.num_of_layers):
                cka_same_layer = cka_matrix[layer][layer]
                client1 = client_pair[0]
                client2 = client_pair[1]
                
                self.layerwise_cka_matrix[layer][client1][client2] = cka_same_layer
                if(client1 != client2):
                    self.layerwise_cka_matrix[layer][client2][client1] = cka_same_layer
    
    def saving_plot(self, save_path: str = None):
        
        for layer, cka_matrix in self.layerwise_cka_matrix.items():

            fig, ax = plt.subplots()
            im = ax.imshow(cka_matrix, origin='upper', cmap='GnBu', vmin=0.0, vmax=1.0)
            ax.set_xlabel("Clients", fontsize=15)
            ax.set_ylabel("Clients", fontsize=15)
        
            ax.set_title(f"Layer{layer+1}", fontsize=18)
      

            add_colorbar(im)
            plt.tight_layout()

            if save_path is not None:
                file_name = save_path + "/layer{}.png".format(layer+1)
                plt.savefig(file_name, dpi=300)


    def print_average(self, save_path: str = None):
        logfile = open(save_path + "/cka_average.txt",'a')
        for layer, cka_matrix in self.layerwise_cka_matrix.items():
            logfile.write("%f"%cka_matrix.mean().item())
            logfile.write("\n")
        logfile.close()
        

