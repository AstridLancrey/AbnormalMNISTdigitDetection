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
import itertools


######################################## FUNCTION ANS CLASS DEFINITION

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
        # out_width = (28+2-5)/2+1 = 27/2+1 = 13
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=0)
        # out_width = (14-5)/2+1 = 5
        #self.drop1=nn.Dropout2d(p=0.3) 
        # 6 * 6 * 16 = 576
        self.linear1 = nn.Linear(3*3*32, 128)
        self.linear2 = nn.Linear(128, latent_dims)
        self.linear3 = nn.Linear(128, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        #self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        #self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        #print(x.shape)
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        #print(x.shape)
        x = torch.flatten(x, start_dim=1)
        #print(x.shape)
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
        

def plot_ae_outputs(encoder,decoder, chiffre_anormal):
    
    n = 20
    plt.figure(figsize=(10,4.5))
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = test_set[i][0].unsqueeze(0).to(device) #### L'eval visuelle se fait sur le test!
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
         rec_img  = decoder(encoder(img))
      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.savefig('./Results/recons_VAE_'+str(chiffre_anormal)+'.jpg' )
    #plt.show()
    
def predict_normalORanomay(pertes, chiffre_anormal, seuil):
    
    y_predicted = []
    for j in pertes: 
      if chiffre_anormal != 1:
        if j > seuil:
           y_predicted.append(0) #anomaly
        else:
           y_predicted.append(1) #normal
      else:
        seuil = 0.035
        if j < seuil:
           y_predicted.append(0) #anomaly
        else:
           y_predicted.append(1) #normal
        
    return y_predicted
    
    
def plot_mse(pertes, y_true, seuil, chiffre_anormal):

    df = pd.DataFrame(y_true)
    df['MSE'] = np.array(pertes)
    df.columns=['classe','MSE']

    plt.figure(figsize=(15,5))
    plt.hist(x=df.MSE[df.classe == 1], bins = 15,
            orientation = 'horizontal', rwidth = 0.8)
    plt.hist(x=df.MSE[df.classe == 0], bins = 15, color = '#EE3459',
            orientation = 'horizontal', rwidth = 0.8)
    plt.axhline(y=seuil, color='black', linestyle='--')
    plt.savefig('./Results/MSE_distrib_'+str(chiffre_anormal)+'.jpg' )
    #plt.show();
    
def affich_conf_matrix(y_true, y_pred, chiffre_anormal):
    
  crosstab = pd.crosstab(pd.Series(y_true), pd.Series(y_pred), 
                         rownames=['Classe réelle'],
                         colnames=['Classe prédite'])

  plt.figure(figsize=(7,5))
  plt.imshow(crosstab, interpolation='nearest',cmap='Blues')
  plt.colorbar()
  classes=['anomaly','normal']
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes)
  plt.yticks(tick_marks, classes)

  for i, j in itertools.product(range(crosstab.shape[0]), range(crosstab.shape[1])):
             plt.text(j, i, np.round(np.array(crosstab)[i, j],2),
             horizontalalignment="center",
             color="white" if np.array(crosstab)[i, j] > ( np.array(crosstab).max() / 2) else "black")

  plt.ylabel('Vrais labels')
  plt.xlabel('Labels prédits')
  plt.title("chiffre anormal: "+str(chiffre_anormal))
  plt.savefig('./Results/ConfMatrix_'+str(chiffre_anormal)+'.jpg' )
  #plt.show();

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
    
    ##### Re initialize model by loading the saved corresponding one
    torch.manual_seed(0)
    d = 4
    lr = 0.001 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
    vae_pathname = './model_files/VAE_'+str(i)+'.pth'
    vae = VariationalAutoencoder(latent_dims=d)
    vae.load_state_dict(torch.load(vae_pathname, map_location=torch.device('cpu')))
    #vae.to(device)

    ################### COMPUTE MSE FOR ALL ORIGINAL/RECONSTITUTIONS to determine an anomaly threshold

    pertes = []
    from sklearn.metrics import mean_squared_error
    from tqdm import tqdm
    for sample in tqdm(test_set):
       image = sample[0].unsqueeze(0).to(device)
       original_image = image.cpu().squeeze().numpy()      
       vae.encoder.eval()
       vae.decoder.eval()
       with torch.no_grad():
         recons_image  = vae.decoder(vae.encoder(image))
         recons_image = recons_image.cpu().squeeze().numpy()
       pertes.append(mean_squared_error(original_image,recons_image))
     
     
    ############### MODEL EVALUATION
    
    # display VAE outputs compared to originals for the 20 1st images of the test set (will store picture in Results Folder)
    plot_ae_outputs(vae.encoder, vae.decoder, i)

    anom_threshold = 0.05
    
    # display mse distributions for normal (blue) and anomaly (pink) files with the dashed line representing the anomaly threshold
    # (will store picture in Results Folder)
    plot_mse(pertes, y_test, anom_threshold, i)
    
    y_predicted = predict_normalORanomay(pertes, i, seuil=anom_threshold)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_predicted)
    print('score AUC {:0.2f} %'.format(auc(fpr, tpr)*100))
    
    # display confusion matrix (will store picture in Results Folder)
    affich_conf_matrix(y_test, y_predicted, i)
    
    
    #################### INITIAL EXERCISE REQUEST
    # compute rocauc
    roc_auc = roc_auc_score(y_test, y_predicted)
    aucs.append(roc_auc)

print("roc_auc per digit:")
print(["{:0.3f} ".format(auc) for auc in aucs])
print("average roc_auc:")
print("{:0.3f}".format(torch.tensor(aucs).mean()))

