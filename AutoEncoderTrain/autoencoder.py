# importing utilities
import os
import sys
from datetime import datetime
from pathlib import Path

# importing data science libraries
import pandas as pd
import random as rd
import numpy as np

# importing pytorch libraries
import torch
from torch import nn
from torch import autograd
from torch.utils.data import DataLoader

pathBase = str(Path.cwd())

#https://github.com/GitiHubi/deepAI/blob/master/GTC_2018_Lab.ipynb
#Model heavily influenced from notebook above 

def Square2Conditional(int, step):
    #512, 256, 128, 64, 32, 16, 8, 4, 2, 1
    #1, 2, 3, 4, 5, 6, 7, 8, 9
    return

#Autoencoder scorer/loss function
#Calculates loss between input and reoconstructured output
#binary-cross-entropy error (BCE)
loss_function = nn.BCEWithLogitsLoss(reduction='mean')

class encoder(nn.Module):
    def __init__(self, numpyData):
        super(encoder, self).__init__()
        # specify layer 1 - in 618, out 512
        #Need to change the layers to consider squres of 2 relative to the initial input features/count of columns
        #initial in_features = the input of original dataset features, as concept of autoencoder is to recreate
        #the provided dataset.
        self.encoder_L1 = nn.Linear(in_features=numpyData.shape[1], out_features=512, bias=True) # add linearity 
        nn.init.xavier_uniform_(self.encoder_L1.weight) # init weights according to [9]
        self.encoder_R1 = nn.LeakyReLU(negative_slope=0.4, inplace=True) # add non-linearity according to [10]

        # specify layer 2 - in 512, out 256
        self.encoder_L2 = nn.Linear(512, 256, bias=True)
        nn.init.xavier_uniform_(self.encoder_L2.weight)
        self.encoder_R2 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # specify layer 3 - in 256, out 128
        self.encoder_L3 = nn.Linear(256, 128, bias=True)
        nn.init.xavier_uniform_(self.encoder_L3.weight)
        self.encoder_R3 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # specify layer 4 - in 128, out 64
        self.encoder_L4 = nn.Linear(128, 64, bias=True)
        nn.init.xavier_uniform_(self.encoder_L4.weight)
        self.encoder_R4 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # specify layer 5 - in 64, out 32
        self.encoder_L5 = nn.Linear(64, 32, bias=True)
        nn.init.xavier_uniform_(self.encoder_L5.weight)
        self.encoder_R5 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # specify layer 6 - in 32, out 16
        self.encoder_L6 = nn.Linear(32, 16, bias=True)
        nn.init.xavier_uniform_(self.encoder_L6.weight)
        self.encoder_R6 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # specify layer 7 - in 16, out 8
        self.encoder_L7 = nn.Linear(16, 8, bias=True)
        nn.init.xavier_uniform_(self.encoder_L7.weight)
        self.encoder_R7 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # specify layer 8 - in 8, out 4
        self.encoder_L8 = nn.Linear(8, 4, bias=True)
        nn.init.xavier_uniform_(self.encoder_L8.weight)
        self.encoder_R8 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # specify layer 9 - in 4, out 3
        self.encoder_L9 = nn.Linear(4, 3, bias=True)
        nn.init.xavier_uniform_(self.encoder_L9.weight)
        self.encoder_R9 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # init dropout layer with probability p
        self.dropout = nn.Dropout(p=0.0, inplace=True)
        
    def GenerateEncoders():
        #Method to generate variable number of encoders based on provided shape of dataset
        return

    def forward(self, x):
        # define forward pass through the network
        x = self.encoder_R1(self.dropout(self.encoder_L1(x)))
        x = self.encoder_R2(self.dropout(self.encoder_L2(x)))
        x = self.encoder_R3(self.dropout(self.encoder_L3(x)))
        x = self.encoder_R4(self.dropout(self.encoder_L4(x)))
        x = self.encoder_R5(self.dropout(self.encoder_L5(x)))
        x = self.encoder_R6(self.dropout(self.encoder_L6(x)))
        x = self.encoder_R7(self.dropout(self.encoder_L7(x)))
        x = self.encoder_R8(self.dropout(self.encoder_L8(x)))
        x = self.encoder_R9(self.encoder_L9(x)) # don't apply dropout to the AE bottleneck
        return x



# implementation of the decoder network
class decoder(nn.Module):
    def __init__(self, numpyData):
        super(decoder, self).__init__()
        # specify layer 1 - in 3, out 4
        self.decoder_L1 = nn.Linear(in_features=3, out_features=4, bias=True) # add linearity 
        nn.init.xavier_uniform_(self.decoder_L1.weight)  # init weights according to [9]
        self.decoder_R1 = nn.LeakyReLU(negative_slope=0.4, inplace=True) # add non-linearity according to [10]

        # specify layer 2 - in 4, out 8
        self.decoder_L2 = nn.Linear(4, 8, bias=True)
        nn.init.xavier_uniform_(self.decoder_L2.weight)
        self.decoder_R2 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # specify layer 3 - in 8, out 16
        self.decoder_L3 = nn.Linear(8, 16, bias=True)
        nn.init.xavier_uniform_(self.decoder_L3.weight)
        self.decoder_R3 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # specify layer 4 - in 16, out 32
        self.decoder_L4 = nn.Linear(16, 32, bias=True)
        nn.init.xavier_uniform_(self.decoder_L4.weight)
        self.decoder_R4 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # specify layer 5 - in 32, out 64
        self.decoder_L5 = nn.Linear(32, 64, bias=True)
        nn.init.xavier_uniform_(self.decoder_L5.weight)
        self.decoder_R5 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # specify layer 6 - in 64, out 128
        self.decoder_L6 = nn.Linear(64, 128, bias=True)
        nn.init.xavier_uniform_(self.decoder_L6.weight)
        self.decoder_R6 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        
        # specify layer 7 - in 128, out 256
        self.decoder_L7 = nn.Linear(128, 256, bias=True)
        nn.init.xavier_uniform_(self.decoder_L7.weight)
        self.decoder_R7 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # specify layer 8 - in 256, out 512
        self.decoder_L8 = nn.Linear(256, 512, bias=True)
        nn.init.xavier_uniform_(self.decoder_L8.weight)
        self.decoder_R8 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # specify layer 9 - in 512, out 618
        self.decoder_L9 = nn.Linear(in_features=512, out_features=numpyData.shape[1], bias=True)
        nn.init.xavier_uniform_(self.decoder_L9.weight)
        self.decoder_R9 = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # init dropout layer with probability p
        self.dropout = nn.Dropout(p=0.0, inplace=True)

    def forward(self, x):
        # define forward pass through the network
        x = self.decoder_R1(self.dropout(self.decoder_L1(x)))
        x = self.decoder_R2(self.dropout(self.decoder_L2(x)))
        x = self.decoder_R3(self.dropout(self.decoder_L3(x)))
        x = self.decoder_R4(self.dropout(self.decoder_L4(x)))
        x = self.decoder_R5(self.dropout(self.decoder_L5(x)))
        x = self.decoder_R6(self.dropout(self.decoder_L6(x)))
        x = self.decoder_R7(self.dropout(self.decoder_L7(x)))
        x = self.decoder_R8(self.dropout(self.decoder_L8(x)))
        x = self.decoder_R9(self.decoder_L9(x)) # don't apply dropout to the AE output    
        return x


#Put autoencoder method here
def FitAutoEncoder(numpyarray):
    #Imports encoder object, which encodes features in several layers
    encoder_train = encoder(numpyarray)
    #Imports decoder object, which decodes/recreates dataset in cooresponding layers of encoder
    decoder_train = decoder(numpyarray)

    #Autoencoder scorer/loss function
    #Calculates loss between input and reoconstructured output
    #binary-cross-entropy error (BCE)
    loss_function = nn.BCEWithLogitsLoss(reduction='mean')
    
    #Rate and optimization
    learning_rate = 1e-3
    encoder_optimizer = torch.optim.Adam(encoder_train.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder_train.parameters(), lr=learning_rate)

    #Training paramters
    num_epochs = 10
    mini_batch_size = 128

    #Iterable torch data loader
    torch_dataset = torch.from_numpy(numpyarray).float()
    dataloader = DataLoader(torch_dataset, batch_size=mini_batch_size, shuffle=True, num_workers=0)

    # init collection of mini-batch losses
    losses = []

    # convert encoded transactional data to torch Variable
    data = autograd.Variable(torch_dataset)

    # train autoencoder model
    for epoch in range(num_epochs):

        # init mini batch counter
        mini_batch_count = 0
        # set networks in training mode (apply dropout when needed)
        encoder_train.train()
        decoder_train.train()
        # start timer
        start_time = datetime.now()

        # iterate over all mini-batches
        for mini_batch_data in dataloader:
            # increase mini batch counter
            mini_batch_count += 1
            # convert mini batch to torch variable
            mini_batch_torch = autograd.Variable(mini_batch_data)
            # =================== (1) forward pass ===================================
            # run forward pass
            z_representation = encoder_train(mini_batch_torch) # encode mini-batch data
            mini_batch_reconstruction = decoder_train(z_representation) # decode mini-batch data
            # =================== (2) compute reconstruction loss ====================
            # determine reconstruction loss
            reconstruction_loss = loss_function(mini_batch_reconstruction, mini_batch_torch)
            # =================== (3) backward pass ==================================
            # reset graph gradients
            decoder_optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            # run backward pass
            reconstruction_loss.backward()
            # =================== (4) update model parameters ========================
            # update network parameters
            decoder_optimizer.step()
            encoder_optimizer.step()

            # =================== monitor training progress ==========================
            # print training progress each 1'000 mini-batches
            if mini_batch_count % 1000 == 0:
                # print the training mode: either on GPU or CPU
                mode = 'GPU' if (torch.backends.cudnn.version() != None) and (USE_CUDA == True) else 'CPU'
                # print mini batch reconstuction results
                now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
                end_time = datetime.now() - start_time
                print('[LOG {}] training status, epoch: [{:04}/{:04}], batch: {:04}, loss: {}, mode: {}, time required: {}'.format(now, (epoch+1), num_epochs, mini_batch_count, np.round(reconstruction_loss.item(), 4), mode, end_time))

                # reset timer
                start_time = datetime.now()
# =================== evaluate model performance =============================

        # set networks in evaluation mode (don't apply dropout)
        encoder_train.cpu().eval()
        decoder_train.cpu().eval()

        # reconstruct encoded transactional data
        reconstruction = decoder_train(encoder_train(data))
        
        # determine reconstruction loss - all transactions
        reconstruction_loss_all = loss_function(reconstruction, data)
                
        # collect reconstruction loss
        losses.extend([reconstruction_loss_all.item()])
        
        # print reconstuction loss results
        now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
        print('[LOG {}] training status, epoch: [{:04}/{:04}], loss: {:.10f}'.format(now, (epoch+1), num_epochs, reconstruction_loss_all.item()))

        #Note, after final save to disk need to import into models dicitionary, and class will be saved to disk instead
        # =================== save model snapshot to disk ============================
        

        # save trained encoder model file to disk
        encoder_model_name = "ep_{}_encoder_model.pth".format((epoch+1))
        encoderSavePath = pathBase + "\\AutoEncoderDict\\" + encoder_model_name

        #torch.save(encoder_train.state_dict(), os.path.join("./models", encoder_model_name))
        torch.save(encoder_train.state_dict(), encoderSavePath)

        # save trained decoder model file to disk
        decoder_model_name = "ep_{}_decoder_model.pth".format((epoch+1))
        decoderSavePath = pathBase + "\\AutoEncoderDict\\" + decoder_model_name
        torch.save(decoder_train.state_dict(), decoderSavePath)
    
    return  