import numpy as np
import pandas as pd
#import matplotlib to plot graphs
import matplotlib.pyplot as plt

#Librabries for creating and training neural Networks
from torch import nn, optim
import torch.nn.functional as F
import torch

def iterate_minibatches(X, y, mini_batchsize, indices):
    '''
    Function yields minibatches for features andtargets for shuffled indices at each epoch and returns ir

    Args:
    ---
    X(tensor)- tensor Features for training
    y(tensor) - tensor Targets for training
    mini_batch(int) - Rows to be returned for each mini batch
    indices(np.array) - shuffled array of indicies

    Returns:
    -------
    Mini batch features and targets
    '''
    
    for start_ind in range(0, X.shape[0], mini_batchsize):
        end_ind = min(start_ind + mini_batchsize, X.shape[0])
        #print("start:end",start_idx,end_ind)
        new_in = indices[start_ind:end_ind]
        #print("new_in",new_in)
        yield X[new_in], y[new_in]
        
        

def create_train_model(X_train,y_train):
    '''
    creates,trains a NN model and returns the model
    Args:
    ---
    X_train(DataFrame)-  Features for training
    y(DataFrame) - Targets for training

    Returns:
    -------
    optimized neural network model
    '''   
    #number input layer noders = total number of columns in X_train
    inputs_ = X_train.shape[1]
    ## using torch nn module to define the structure of nueral network
    class NeuralNet(nn.Module):
        def __init__(self):
            #initializing using nn __init__()
            super().__init__()
            self.fc1 = nn.Linear(inputs_, 307)
            self.fc2 = nn.Linear(307, 256)
            self.fc3 = nn.Linear(256, 8)
            self.dropout = nn.Dropout(p=0.2)
        def forward(self, x):
            #print(x.shape)
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.dropout(F.relu(self.fc2(x)))
            x = F.log_softmax(self.fc3(x), dim=1)

            return x
        
    
    model1 = NeuralNet()
    # Defining the loss
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model1.parameters(), lr=0.005)
    epochs = 450 #number of times the whole training data is feeded in the neural network for training
    val_print = 4 #validate after every 4rth time and print training, validation score and validation accuracy
    batch_size = 9900 #for this 1/4rth of the X_train rows i.e 39,577 ~ 39,600 
    step = 0
    
    #converting to torch values for training
    X_train_t = torch.tensor(X_train.values).type(torch.float)
    y_train_t = torch.tensor(y_train['rate_color'].values).type(torch.long)
    
    #indices in X_train
    indices = np.arange(X_train_t.shape[0])
    
    #stores respective losses
    train_losses, val_losses = [], []
    loss_counts = []
    loss_count = 0
    #for epochs from 0 to n-1 epochs
    for e in range(epochs):
        #shuffle indices after every epoch to create new shuffled mini batches
        np.random.shuffle(indices)
        #setting running loss as 0
        if step == 0:
            running_loss = 0
            
        #yield mini batch till the last of the row is reached
        for batch in iterate_minibatches(X_train_t, y_train_t, batch_size,indices):
            step += 1
            X_batch, y_batch = batch
            #if first 3 mini batch, train the model
            if step%val_print != 0: 
                #print("train")
                optimizer.zero_grad()

                log_ps = model1(X_batch)
                #calculate loss
                loss = criterion(log_ps, y_batch)
                #print(loss)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                #print(running_loss)
                
            # for the 4rth mini batch validate
            if step%val_print == 0:
                #print("validate")
                loss_count +=1
                val_loss = 0
                accuracy = 0

                # gradients turned off for validation
                with torch.no_grad():
                    model1.eval()
                    log_ps = model1(X_batch)
                    val_loss += criterion(log_ps, y_batch)

                    ps = torch.exp(log_ps)
                    #gets the top predicted class
                    top_p, top_class = ps.topk(1, dim=1)
                    #binaries into true and false by comparing y_test and predicted list
                    equals = top_class == y_batch.view(*top_class.shape)
                    #calcuates the accuracy 
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

                #next forward feed
                model1.train()
                
                # calculating average training loss of the three trains in the epochs
                #calculating validation loss for the 4rth mini batch
                train_losses.append(running_loss/(val_print- 1))
                val_losses.append(val_loss)
                loss_counts.append(loss_count)
                running_loss = 0
                
                #print losses,accuracy after every 4rth epoch
                if e%5 == 1:
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(train_losses[-1]),
                          "validation Loss: {:.3f}.. ".format(val_losses[-1]),
                          "validation Accuracy: {:.3f}".format(accuracy))
    
    #Print the learning curve Validation score  and training score
    plt.plot(loss_counts[10:],train_losses[10:],'-b', label='train')
    plt.plot(loss_counts[10:],val_losses[10:],'-g', label='validation')
    
    #return the trained model
    return model1