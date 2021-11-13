import os
from comet_ml import Experiment
import sys
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader import FaceLoader
from model import VAE

# Dataloader parameters
train_params = {'batch_size': 128,
          'shuffle': False,
          'drop_last': False,
          'num_workers': 2}
test_params = {'batch_size': 128,
          'shuffle': False,
          'drop_last': False,
          'num_workers': 2}

def train(model, optimizer, train_loader, recon_crit, class_crit, kl_weight):
    
    # Set model to train and get total loss
    total_loss = 0.0
    total_accuracy = 0.0
    model = model.train()

    for (batch, data) in tqdm(enumerate(train_loader)):

        # Get data
        img = data['img'].float()
        lab = data['lab']

        # Make sure image is between 0 and 1 (We arent normalising the input because this is just a test)
        img = img.permute((0,3,1,2))/255
        
        # Zero the gradients
        optimizer.zero_grad()

        # Get the outputs from the model
        enc_out, dec_out = model(img)
        mu,cov,face = enc_out[1], enc_out[2], enc_out[3]

        # Classify the face
        preds = torch.argmax(face,dim=1)
        
        # Compute all the 3 losses
        recon_loss = recon_crit(dec_out,img)
        kl_loss = 1/2 * torch.sum(torch.exp(cov) + torch.pow(mu,2) - 1 - cov)
        vae_loss = kl_weight * kl_loss + recon_loss
        class_loss = class_crit(face, lab)

        # Get the accuracy of classification
        accuracy = get_accuracy(preds, lab)
        
        # Get total loss
        loss = class_loss + vae_loss
        total_loss += loss
        total_accuracy += accuracy
        
        # Compute gradients and back prop
        loss.backward()
        optimizer.step()
        
    
    return model, optimizer, total_loss/(batch+1), total_accuracy/(batch+1)

def test(model, test_loader, recon_crit, class_crit, kl_weight, name2id, epoch, experiment = None):
    
    # Make sure model is in eval mode
    total_loss = 0.0
    total_accuracy = 0.0
    model = model.eval()

    # We dont want to compute gradients in test
    with torch.no_grad():
        for (batch, data) in tqdm(enumerate(test_loader)):

            # Get data
            img = data['img'].float()
            lab = data['lab']
            img = img.permute((0,3,1,2))/255
            
            # Get outputs from model
            enc_out, dec_out = model(img)
            mu,cov,face = enc_out[1], enc_out[2], enc_out[3]

            # Classify the face
            preds = torch.argmax(face,dim=1)
            
            # To check reconstruction
            if epoch % 10 == 0:
                plot_preds(img,dec_out, lab, preds, batch, name2id, epoch,experiment)
            
            # Get the accuracy of classification
            accuracy = get_accuracy(preds, lab)

            # Compute test loss
            recon_loss = recon_crit(dec_out,img)
            kl_loss = 1/2 * torch.sum(torch.exp(cov) + torch.pow(mu,2) - 1 - cov)
            vae_loss = kl_weight * kl_loss + recon_loss
            class_loss = class_crit(face, lab)
            
            # Store metrics
            loss = class_loss + vae_loss
            total_loss += loss
            total_accuracy += accuracy
        
        return total_loss/(batch+1), total_accuracy/(batch+1)

def plot_preds(img,dec_out, lab, preds, batch, name2id, epoch, experiment):
    for id_, elem in enumerate(img):
        
        fig = plt.figure()
        ax = fig.add_subplot(1,2,1)
        ax.imshow(elem.permute((1,2,0)).numpy())
        ax.set_title(name2id[str(lab[id_].item())])
        ax = fig.add_subplot(1,2,2)
        ax.imshow(dec_out[id_].permute((1,2,0)).numpy())
        ax.set_title(name2id[str(preds[id_].item())])
        if experiment is not None:
            experiment.log_figure(figure_name=str(batch)+"-"+str(id_),figure=fig,step=epoch)
        fig.savefig("./reconstructions/"+str(batch)+"-"+str(id_)+".png")
        plt.close('all')

def get_accuracy(preds, lab):
    same = (preds == lab)
    return torch.sum(same)/len(lab)

if __name__ == '__main__':

    with open("config.json",'r') as f:
        config = json.load(f)
        f.close()
    
    # Create Comet Experiment
    if config['LOG-COMET']:
        experiment = Experiment(
            api_key=config['api-key'],
            project_name=config['project_name'],
            workspace=config['workspace'],
        )
    else:
        experiment = None


    dataset_path = "./data/"
    # Instantiate data loaders
    train_dataset = FaceLoader(dataset_path = dataset_path, loader_type="Train")
    test_dataset = FaceLoader(dataset_path = dataset_path, loader_type="Test")
    
    train_loader = DataLoader(train_dataset, **train_params)
    test_loader = DataLoader(test_dataset, **test_params)

    # Get the id to name mapping
    with open(dataset_path + 'name2id.json', 'r') as f:
        name2id = json.load(f)
        f.close()

    # Instantiate the model
    model = VAE(num_classes=len(list(name2id.keys())))

    # Instantiate the optimizer
    optimizer = optim.Adam(model.parameters(), lr=3e-4,
                           weight_decay=1e-3)
    
    # Set the loss functions
    recon_crit = nn.L1Loss(reduction='mean')
    kl_weight = 0.0005

    # We want to weight the CE Loss because there is a huge class imbalance
    nums = np.load(dataset_path + "num_faces.npy")
    percentages = nums/np.sum(nums)
    weights = torch.FloatTensor([(1-p) for p in percentages])
    class_crit = nn.CrossEntropyLoss(weight = weights)
    

    NUM_EPOCH=1000

    # Run training and testing
    for epoch in range(NUM_EPOCH):
        model, optimizer, train_loss, train_acc = train(model, optimizer, train_loader, recon_crit, class_crit, kl_weight)
        if config['LOG-COMET']:
                experiment.log_metrics({'Train Loss': train_loss.item()}, epoch=epoch)
                experiment.log_metrics({'Train Accuracy': train_acc.item()}, epoch=epoch)
        print("------- ",epoch,": Train loss: ", train_loss.item(), "Accuracy: ", train_acc.item()," -------")

        test_loss, test_acc = test(model, test_loader, recon_crit, class_crit, kl_weight,name2id, epoch, experiment)
        if config['LOG-COMET']:
                experiment.log_metrics({'Test Loss': test_loss.item()}, epoch=epoch)
                experiment.log_metrics({'Test Accuracy': test_acc.item()}, epoch=epoch)

        
        print("------- ",epoch,": Test loss: ", test_loss.item(),"Accuracy: ", test_acc.item()," -------")

        