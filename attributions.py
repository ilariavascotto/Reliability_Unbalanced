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


parser = argparse.ArgumentParser(description="attributions")
parser.add_argument("--neigh", type=str, default="medoid")
args = parser.parse_args()
args.seed = np.random.randint(1)
print(args)


# The data files cannot be shared as they are proprietary
Xtrain = torch.load(os.path.join(os.getcwd(), "data", "Xtrain.pt"))
Xvalid = torch.load(os.path.join(os.getcwd(), "data", "Xvalid.pt"))
Xtest = torch.load(os.path.join(os.getcwd(), "data", "Xtest.pt"))

ytrain = torch.load(os.path.join(os.getcwd(), "data", "ytrain.pt"))
yvalid = torch.load(os.path.join(os.getcwd(), "data", "yvalid.pt"))
ytest = torch.load(os.path.join(os.getcwd(), "data", "ytest.pt"))

# The model cannot be shared
net_frost = net.network()
net_frost.load_state_dict(torch.load(os.path.join(os.getcwd(), "model_gamma_25_alpha_75.pt")))


if args.neigh=="medoid":
    folder = os.path.join(os.getcwd(), "results")
else:
    folder = os.path.join(os.getcwd(), "results_random")

if not os.path.isdir(folder):
    os.makedirs(folder)
    print(f"New folder created at {folder}")


if args.neigh == "medoid":

    kmedoids_folder = os.path.join(folder, "kmedoids")
    if not os.path.isdir(kmedoids_folder):
        os.mkdir(kmedoids_folder)
        print(f"New folder created for kmedoids")   

    kmedoids_file = os.path.join(kmedoids_folder, "kmedoids.joblib")
    centers_path = os.path.join(kmedoids_folder, "centers.npy")
    knn_overall_path = os.path.join(kmedoids_folder, f"knn_overall_5.npy")
    labels_validation_path = os.path.join(kmedoids_folder, "labels_validation.npy")
    labels_test_path = os.path.join(kmedoids_folder, f"labels_test.npy")

    kmedoids = joblib.load(kmedoids_file)
    cluster_centers = np.load(centers_path)
    knn_overall = np.load(knn_overall_path)
    labels = np.load(labels_validation_path)
    labels_test = np.load(labels_test_path)


print("Validation set")
xx = Xvalid.detach().numpy()
yy = net_frost(Xvalid).detach().numpy()>= 0.5

num_pt, num_ft = xx.shape

num = 100
neigh_size = []
results_valid = np.zeros(shape = (num_pt, num+1, num_ft, 3))
distances = np.zeros(shape=(num_pt, num))

for i in range(num_pt):
    if (i%1000)==0:
        print(f"{i+1}/{num_pt}")

    if args.neigh =="medoid":
        x = ut.medoid_neighbourhood(xx[i, :], i, labels, 100, knn_overall, cluster_centers, alpha = 0.05, alpha_cat=0, categorical_features=[], discrete=[], ordinal = [], categorical_names={}, num_ft_or=num_ft)
    else:
        x = ut.random_neighbourhood(xx[i,:], 100, sigma= 0.05, gamma_cat = 0, categorical_features=[], categorical_names={}, num_ft_or=num_ft)

    
    x = ut.keep_neighbourhood(x, target = yy[i], model = net_frost, encoder=None)

    dim_neigh = x.shape[0]
    neigh_size.append(dim_neigh)

    distances[i,:(dim_neigh-1)] = np.linalg.norm(x[0,:]-x[1:,:], axis=1)

    
    if dim_neigh == 1:
        print(f"No points in the neighbourhood for test point {i}")
        continue

    attr = ut.compute_attributions(net_frost, x, target=0, lrp_rule=GammaRule)

    results_valid[i, :dim_neigh, :, :] = attr[:, :, :]
    

folder_type = os.path.join(folder, "Validation")
if not os.path.isdir(folder_type):
    os.mkdir(folder_type)
    print(f"New folder created for validation set")


np.save(os.path.join(folder_type, "attributions"), results_valid) #num_points x (num+1) x num_ft_or x 3
np.save(os.path.join(folder_type, "neigh_size"), neigh_size) #num_points
np.save(os.path.join(folder_type, "distances"), distances) #num_points x 100
print("Saved attributions successfully.")

robustness = np.zeros(shape = (num_pt, 3))

for i in range(num_pt):
    dim_neigh = neigh_size[i]

    if dim_neigh == 1:
        continue

    for j in range(3):
        rho_ = rho(results_valid[i, 0, :, j], results_valid[i, 1:dim_neigh, :, j], axis=1).correlation
        robustness[i,j] = np.mean(rho_)
    
np.save(os.path.join(folder_type, f"robustness"), robustness)




print("Test set")
xx = Xtest.detach().numpy()
yy = net_frost(Xtest).detach().numpy()>= 0.5

num_pt, num_ft = xx.shape
num = 100
neigh_size = []
results_test = np.zeros(shape = (num_pt, num+1, num_ft, 3))
distances = np.zeros(shape=(num_pt, num))

for i in range(num_pt):
    if (i%1000)==0:
        print(f"{i+1}/{num_pt}")
       
    if args.neigh =="medoid":
        x = ut.medoid_neighbourhood(xx[i, :], i, labels_test, 100, knn_overall, cluster_centers, alpha = 0.05, alpha_cat=0, categorical_features=[], discrete=[], ordinal = [], categorical_names={}, num_ft_or=num_ft)
    else:
        x = ut.random_neighbourhood(xx[i,:], 100, sigma= 0.05, gamma_cat = 0, categorical_features=[], categorical_names={}, num_ft_or=num_ft)

    
    x = ut.keep_neighbourhood(x, target = yy[i], model = net_frost, encoder=None)

    dim_neigh = x.shape[0]
    neigh_size.append(dim_neigh)

    distances[i,:(dim_neigh-1)] = np.linalg.norm(x[0,:]-x[1:,:], axis=1)

    
    if dim_neigh == 1:
        print(f"No points in the neighbourhood for test point {i}")
        continue

    attr = ut.compute_attributions(net_frost, x, target=0, lrp_rule=GammaRule)

    results_test[i, :dim_neigh, :, :] = attr[:, :, :]
    

folder_type = os.path.join(folder, "Test")
if not os.path.isdir(folder_type):
    os.mkdir(folder_type)
    print(f"New folder created for test set")


np.save(os.path.join(folder_type, "attributions"), results_test) #num_points x (num+1) x num_ft_or x 3
np.save(os.path.join(folder_type, "neigh_size"), neigh_size) #num_points
np.save(os.path.join(folder_type, "distances"), distances) #num_points x 100

print("Saved attributions successfully.")

robustness = np.zeros(shape = (num_pt, 3))

for i in range(num_pt):
    dim_neigh = neigh_size[i]

    if dim_neigh == 1:
        continue

    for j in range(3):
        rho_ = rho(results_test[i, 0, :, j], results_test[i, 1:dim_neigh, :, j], axis=1).correlation
        robustness[i,j] = np.mean(rho_)
    
np.save(os.path.join(folder_type, f"robustness"), robustness)