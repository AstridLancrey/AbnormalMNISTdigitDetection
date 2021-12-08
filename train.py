import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.distributions import Normal

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=0)
        self.linear1 = nn.Linear(3*3*32, 128)
        self.linear2 = nn.Linear(128, latent_dims)
        self.linear3 = nn.Linear(128, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # COMMENTER LES 2 LIGNES SUIVANTES SI ON UTILISE UN CPU
        self.N.loc = self.N.loc.cuda() 
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
        
class Decoder(nn.Module):
    
    def __init__(self, latent_dims):
        super().__init__()

        ### Linear section
        self.decoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(latent_dims, 128),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        ### Unflatten
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        ### Convolutional section
        self.decoder_conv = nn.Sequential(
            # First transposed convolution
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # Second transposed convolution
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            # Third transposed convolution
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )
        
    def forward(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Unflatten
        x = self.unflatten(x)
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        # Apply a sigmoid to force the output to be between 0 and 1 (valid pixel values)
        x = torch.sigmoid(x)
        return x        
        
        
        
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        return self.decoder(z)      


### Training function
def train_epoch(vae, device, dataloader, optimizer):
    # Set train mode for both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for x, _ in dataloader: 
        # Move tensor to the proper device
        x = x.to(device)
        x_hat = vae(x)
        # Evaluate loss
        loss = ((x - x_hat)**2).sum() + vae.encoder.kl

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()

    return train_loss / len(dataloader.dataset)
    
    
def plot_loss(history_train, chiffre_anormal):
    
    plt.figure(figsize=(10,8))
    plt.semilogy(history_train, label='Train')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.show()    
    
#################################################################################
################################### MAIN ########################################
#################################################################################

from dataset import MnistAnomaly

aucs = []
for i in range(10): 
    abnormal_digit = [i] 
    train_set = MnistAnomaly(root=".", train=True, transform=transforms.ToTensor(), 
                             anomaly_categories=abnormal_digit) 

    test_set = MnistAnomaly(root=".", train=False, transform=transforms.ToTensor(),
                            anomaly_categories=abnormal_digit)
    
    y_test = [1 if c == True else 0 for c in test_set.targets]
    print("chiffre anormal: ", i)
    
    train_transform = transforms.Compose([transforms.ToTensor(),])
    test_transform = transforms.Compose([transforms.ToTensor(),])
    train_set.transform = train_transform
    test_set.transform = test_transform
    
    batch_size=256
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    ##### Re initialize model 
    torch.manual_seed(0)
    d = 4
    #vae_pathname = './model_files_examinateur/VAE_'+str(i)+'.pth'
    vae = VariationalAutoencoder(latent_dims=d)
    lr = 0.001 
    optim = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')
    vae.to(device) 
    
    num_epochs = 30
    history={'train_loss':[]}
    for epoch in range(num_epochs):
      train_loss = train_epoch(vae,device,train_loader,optim)
      print('\n EPOCH {}/{} \t train loss {}'.format(epoch + 1, num_epochs,train_loss))
      history['train_loss'].append(train_loss)     
    #plot_loss(history['train_loss'], i)
                  
    #torch.save(vae.state_dict(), vae_pathname)
    

        