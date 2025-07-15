import pandas as pd
import fastparquet
import numpy as np
import xgboost

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

import torch
import joblib

import captum
from lrp import LRP
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule, Alpha1_Beta0_Rule
from scipy.stats import spearmanr as rho


from kmedoids import KMedoids

import net
import utils as ut

import argparse
import os
import warnings
warnings.filterwarnings('ignore')

# The data files cannot be shared as they are proprietary
Xvalid = torch.load(os.path.join(os.getcwd(), "data", "Xvalid.pt"))
Xtest = torch.load(os.path.join(os.getcwd(), "data", "Xtest.pt"))

num_methods = 3

# The model cannot be shared

net_frost = net.network()
net_frost.load_state_dict(torch.load(os.path.join(os.getcwd(), "model_gamma_25_alpha_75.pt")))

folder = os.path.join(os.getcwd(), "results")
folder_type = os.path.join(folder, "Validation")
print("Validation sets")
attributions = np.loadt(os.path.join(folder_type, "attributions.npy"))
neigh_size = np.load(os.path.join(folder_type, "neigh_size.npy"))

num = np.max(neigh_size)

num_points, num_ft_or = Xvalid.shape

aggregation = np.zeros(shape=(num_points, num, num_ft_or))
for idx in range(num_points):
    dim_neigh = neigh_size[idx]
    for i in range(dim_neigh):
        aggregation[idx, i, :], _ = ut.ensemble(attributions[idx, i, : ,:])

attributions = -np.abs(attributions)

np.save(os.path.join(folder, "ensemble"), aggregation)
print("Ensemble computed")

robustness = np.zeros(shape = (num_points, num_methods +1))

for i in range(num_points):
    dim_neigh = neigh_size[i]

    if dim_neigh == 1:
        continue

    for j in range(num_methods):
        rho_ = rho(attributions[i, 0, :, j], attributions[i, 1:dim_neigh, :, j], axis=1).correlation
        robustness[i,j] = np.mean(rho_)
    
    rho_ = rho(aggregation[i,0,:], aggregation[i, 1:dim_neigh, :], axis=1).correlation
    robustness[i, 3] = np.mean(rho_)


np.save(os.path.join(folder_type, f"robustness_ensemble"), robustness)

####################################################################################

folder_type = os.path.join(folder, "Test")
print("Test set")

num_points, num_ft_or = Xtest.shape
attributions = np.loadt(os.path.join(folder_type, "attributions.npy"))
neigh_size = np.load(os.path.join(folder_type, "neigh_size.npy"))

num = np.max(neigh_size)

aggregation = np.zeros(shape=(num_points, num, num_ft_or))
for idx in range(num_points):
    dim_neigh = neigh_size[idx]
    for i in range(dim_neigh):
        aggregation[idx, i, :], _ = ut.ensemble(attributions[idx, i, : ,:])

attributions = -np.abs(attributions)

np.save(os.path.join(folder, "ensemble"), aggregation)
print("Ensemble computed")

robustness = np.zeros(shape = (num_points, num_methods +1))

for i in range(num_points):
    dim_neigh = neigh_size[i]

    if dim_neigh == 1:
        continue

    for j in range(num_methods):
        rho_ = rho(attributions[i, 0, :, j], attributions[i, 1:dim_neigh, :, j], axis=1).correlation
        robustness[i,j] = np.mean(rho_)
    
    rho_ = rho(aggregation[i,0,:], aggregation[i, 1:dim_neigh, :], axis=1).correlation
    robustness[i, 3] = np.mean(rho_)


np.save(os.path.join(folder_type, f"robustness_ensemble"), robustness)