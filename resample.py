import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import scipy.io as sio
from scipy.fft import fft,fftfreq
import time
from Autoencoder import AutoEncoderConv1D
from torch.nn import functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import datetime
import pickle
import sklearn.metrics as metrics

tag = "ThreeDay2"


#test_dam = pickle.load("dam5_seperated.pkl")
with open('dam4_seperated.pkl', 'rb') as f:
    # Load the contents of the file into a variable using pickle.load()
    data1 = pickle.load(f)

with open('dam5_seperated.pkl', 'rb') as f:
    # Load the contents of the file into a variable using pickle.load()
    data2 = pickle.load(f)

data = np.vstack([data1['data'],data2['data']])
date = np.hstack([data1["time"],data2["time"]])

start_date = datetime.datetime(2021,8,23,0,0)
end_date = datetime.datetime(2021,8,30,0,0)




ae_coef = [[] for i in range(13)]
pca_coef = []

train_start = start_date
train_end = start_date + datetime.timedelta(days=1)

test_start = start_date + datetime.timedelta(days=1)
test_end = start_date + datetime.timedelta(days=4)

## ONE DAY
for i in range(8):



    if train_end > end_date or train_start>end_date:
        break

    if test_start >= end_date:
        break

    elif test_end > end_date:
        test_end = end_date
    print("====================================================")
    print("current train data: " + str(train_start) + " to " + str(train_end))
    print("current test data: " + str(test_start) + " to " + str(test_end))

    train_data = data[np.where(date>=train_start)[0][0]:np.where(date>=train_end)[0][0]]
    test_data = data[np.where(date >= test_start)[0][0]:np.where(date >= test_end)[0][0]]

    train_data = np.vstack([train_data,test_data[::2]])

    ori_signal = train_data  # np.vstack([dam1, dam2[0:333]])
    l = ori_signal.shape[0]

    toTensor_sig = torch.tensor(ori_signal)

    train_data_set = TensorDataset(toTensor_sig)

    lr = 0.0001
    batch_size = 32
    mem_dim = 3
    shrink_thres = 0.0025
    Epoch = 50

    MemAE = AutoEncoderConv1D(mem_dim=2000, shrink_thres=shrink_thres)
    MemAE = MemAE.float()

    tr_recon_loss_func = nn.MSELoss()  # is the cross entropy resonable? what about test stage?
    tr_optimizer = torch.optim.Adam(MemAE.parameters(), lr=lr)

    tr_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)

    learning_curve = np.zeros(Epoch)

    models = []

    "save model in 1,3,5,7,10,,12,15,17,20,25,30,40,50 epoch"
    for epoch_idx in range(Epoch):
        print("Epoch: ", epoch_idx + 1)
        cur_MSE = 0
        for batch_idx, cur_data in enumerate(tr_data_loader):
            recon_res = MemAE(cur_data[0].float())  # data is single signal or single signal mulply the batch size?
            recon_sig = recon_res['output']
            loss = tr_recon_loss_func(recon_sig, cur_data[0].float())
            recon_loss_val = loss.item()
            loss = loss
            loss_val = loss.item()
            tr_optimizer.zero_grad()
            loss.backward()
            tr_optimizer.step()
            cur_MSE += loss_val
        learning_curve[epoch_idx] = cur_MSE / batch_size  # really should be divided by batch size????\
        if epoch_idx + 1 == 1 or epoch_idx + 1 == 3 or epoch_idx + 1 == 5 or epoch_idx + 1 == 7 or epoch_idx + 1 == 10 or epoch_idx + 1 == 12 or epoch_idx + 1 == 15 or epoch_idx + 1 == 17 or epoch_idx + 1 == 20 or epoch_idx + 1 == 25 or epoch_idx + 1 == 30 or epoch_idx + 1 == 40 or epoch_idx + 1 == 50:
            torch.save(MemAE, 'online_detection' + '_' + str(epoch_idx + 1) + '_' + tag + '.pth')
            print("model saved, epoch: " + str(epoch_idx + 1))
            models.append(torch.load('online_detection' + '_' + str(epoch_idx + 1) + '_' + tag + '.pth'))



    """PCA coef"""
    #pca_coef = np.zeros(test_data.shape)
    pca_components = 15
    pca = PCA(n_components=pca_components)
    pca.fit(test_data[i:])
    transformed_data = pca.transform(test_data)
    reconstructed_data = pca.inverse_transform(transformed_data)
    for i in range(reconstructed_data.shape[0]):
        pca_coef.append((np.corrcoef(reconstructed_data[i], test_data[i])[0][1]))

    """ Autoencoder Coef"""

    test_normal = torch.tensor(test_data)
    normal_dataset = TensorDataset(test_normal)
    normal_data_loader = DataLoader(normal_dataset, batch_size=25, shuffle=False)
    recon_err_nor = np.zeros(test_normal.size(0))
    encode_normal = np.zeros((test_data.shape[0], 15))
    decoder_in_normal = np.zeros((test_data.shape[0], 15))
    recon_signal_nor = np.zeros((13, test_data.shape[0], 2000))

    print("=====test on normal data=====")
    for batch_idx, cur_data in enumerate(normal_data_loader):
        # print(data[0].size())
        for i in range(len(models)):
            recon_res = models[i](cur_data[0].float())
            recon_sig = recon_res['output']
            recon_signal_nor[i][batch_idx * 25: (batch_idx + 1) * 25] = recon_sig.detach().numpy()


    for i in range(len(models)):
        for j in range(recon_signal_nor.shape[1]):
            ae_coef[i].append(np.corrcoef(recon_signal_nor[i, j], test_normal[j])[0][1])

    train_start = train_start + datetime.timedelta(days=3)
    train_end = train_end + datetime.timedelta(days=3)

    test_start = test_start + datetime.timedelta(days=3)
    test_end = test_end + datetime.timedelta(days=3)

ae_coef = np.array(ae_coef)
pca_coef = np.array(pca_coef)

start_index = np.where(date>=datetime.datetime(2021,8,24,0,0))[0][0]
end_index = np.where(date>=datetime.datetime(2021,8,30,0,0))[0][0]
true_label = np.zeros(data[start_index:end_index].shape[0])
normal_end = np.where(date[start_index:end_index] == data1['time'][-1])[0][0]
true_label[normal_end:] = 1

filtered_auc = np.zeros(13)
for j in range(len(models)):
    filtered_pca = pca_coef[np.where(pca_coef >= 0.99)]
    filter_ae = ae_coef[j][np.where(pca_coef >= 0.99)]
    pred_prob = filtered_pca / np.median(filtered_pca) - filter_ae / np.median(filter_ae)
    fpr, tpr, threshold = metrics.roc_curve(true_label[np.where(pca_coef >= 0.99)], pred_prob, pos_label=1)
    filtered_auc[j] = metrics.auc(fpr, tpr)

np.save(tag+".npy",filtered_auc)

plt.plot(filtered_auc)
#torch.save(MemAE,'auc_roc_dam4_50_2d_mixed.pth')