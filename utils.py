import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from  torch.utils.data import TensorDataset, DataLoader

import captum
from captum.attr import IntegratedGradients, DeepLift
from lrp import LRP
from lrp import linear_rule, non_linear_support
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule, Alpha1_Beta0_Rule


def compute_attributions(model, dataset, target, lrp_rule):
    ig = IntegratedGradients(model)
    dl = DeepLift(model)
    lrp = LRP(model)

    _ = linear_rule(lrp_rule)
    _ = non_linear_support(nn.Sigmoid)

    dataset.requires_grad_()

    METHODS = [ig, dl, lrp]

    attr = np.zeros(shape=(dataset.shape[0], dataset.shape[1], len(METHODS))) #num points x num features x num methods

    for i, method in enumerate(METHODS):
        tmp = method.attribute(dataset, target=target)
        tmp = tmp.detach().numpy()

        if i ==(len(METHODS)-1): #for lrp
            output_score = model(dataset).detach().numpy()#[:, target]
            tmp = tmp* output_score#[:,np.newaxis]

        attr[:, :, i] = tmp
    
    return attr

def ensemble_weights(attr):
    return np.divide(np.std(attr), np.sqrt(np.abs(np.multiply(np.mean(attr), attr))))

def ensemble(reverse_attr, lambda_pen = 0.15):
    num_methods = reverse_attr.shape[1]
    
    weights = [ensemble_weights(attr) for attr in reverse_attr.T] 
    sign = [[1 if el >=0 else 0 for el in attr] for attr in reverse_attr.T]
    rank = np.argsort(np.argsort(-np.abs(reverse_attr), axis=0), axis=0)
    
    idx = np.where(np.std(reverse_attr, axis=0)==0)[0]    
    if len(idx)>0:
        for i in idx:
            rank[:,i] = np.full(fill_value = len(reverse_attr[:,i])-1, shape=reverse_attr[:,i].shape)
            weights[i] = np.full(fill_value = 1/1e-10, shape=reverse_attr[:,i].shape)
            sign[i] = [1 for i in range(len(reverse_attr[:,i]))]
    
    new_ensemble = np.zeros(reverse_attr.shape[0])
    sign=np.sum(sign, axis=0)
    penalization = [1 + lambda_pen*min(sign[i], num_methods-sign[i]) for i in range(sign.shape[0])]

    for i in range(num_methods):
        new_ensemble += np.multiply(rank[:,i], weights[i])
    
    new_ensemble = np.divide(new_ensemble, np.sum(weights, axis=0))
    new_ensemble = np.multiply(new_ensemble, penalization)

    sign = [0 if el<(num_methods+1)/2 else 1 for el in sign]
    return new_ensemble, sign


def keep_neighbourhood(neighbourhood, target, model, encoder):
    if encoder != None:
        try:
            nbh_tensor = torch.Tensor(encoder.transform(neighbourhood).toarray())
        except:
            nbh_tensor = torch.Tensor(encoder.trasform(neighbourhood))
    else:
        nbh_tensor = torch.Tensor(neighbourhood)

    predictions = model(nbh_tensor).detach().numpy() >= 0.5
    #np.argmax(model(nbh_tensor).detach().numpy(), axis=1)
    keep = np.where(target==predictions)[0]
    return nbh_tensor[keep]

def change_categorical_vars(data, idx, categorical_names):

    values = [i for i in range(len(categorical_names[idx]))]
    y = np.random.choice(values, len(data))

    odd = np.where(data ==y)[0]
    if len(odd)>0:
        for el in odd:
            weights = [1./(len(values)-1) if _ != data[el] else 0 for _ in values]
            y[el] = np.random.choice(values, 1, weights)
    
    return y
#################################################################################################################

def random_perturbation(neighbourhood, sigma, gamma_cat, categorical_features, categorical_names, num_ft_or):
    all = np.arange(num_ft_or)
    numerical = [i for i in all if i not in categorical_features]

    mu_ = np.zeros(shape=len(numerical))
    sigma_ = np.diag(np.full(shape=len(numerical), fill_value=sigma))

    try:
        neighbourhood[:, numerical] = neighbourhood[:, numerical] + np.random.multivariate_normal(mean = mu_, cov = sigma_, size=neighbourhood.shape[0])
    except:
        pass

    for idx in categorical_features:
        CAT = np.random.random(size = neighbourhood.shape[0])
        g_cat = np.where(CAT<gamma_cat)[0]

        neighbourhood[g_cat, idx] = change_categorical_vars(neighbourhood[g_cat, idx], idx, categorical_names)

    return neighbourhood


def random_neighbourhood(point, num, sigma, gamma_cat, categorical_features, categorical_names, num_ft_or):
    x = np.full(shape= (num+1, num_ft_or), fill_value=point)

    x[1:,:] = random_perturbation(x[1:,:], sigma, gamma_cat, categorical_features, categorical_names, num_ft_or)

    return x

#################################################################################################################


def change_ordinal_vars(data, idx, categorical_names):
    max_val = len(categorical_names[idx]) -1
    y = np.random.choice([1,-1], len(data))

    y = y+data
    maximum = np.where(y>max_val)[0]
    minimum = np.where(y<0)[0]

    y[maximum] = y[maximum]-2
    y[minimum] = y[minimum]+2
    return y

def medoid_perturbation(neighbourhood, knn_points, centers, alpha, alpha_cat, categorical_features, discrete, ordinal, categorical_names, num_ft_or):
    i = np.random.choice(sorted(knn_points), neighbourhood.shape[0]).astype(int)
    sampled_nn = centers[i]

    ALPHA = np.random.beta(a = alpha*100, b = (1-alpha)*100, size = neighbourhood.shape[0])
    ALPHA = ALPHA[:, np.newaxis]

    all = np.arange(num_ft_or)
    numerical = [i for i in all if i not in categorical_features if i not in discrete]

    neighbourhood[:, numerical] = np.multiply((1-ALPHA), neighbourhood[:, numerical]) + np.multiply(ALPHA, sampled_nn[:, numerical])

    for idx in discrete:
        CAT = np.random.random(neighbourhood.shape[0])
        a_disc = np.where(CAT < alpha)[0]

        neighbourhood[a_disc, idx] = sampled_nn[a_disc, idx]
    
    for idx in categorical_features:
        CAT = np.random.random(size = neighbourhood.shape[0])
        a_cat = np.where(CAT < alpha_cat)[0]

        equal = (neighbourhood[a_cat, idx] == sampled_nn[a_cat, idx])
        change = np.where(equal == 1)[0]
        no_change = np.where(equal==0)[0]

        change_idx = a_cat[change]
        no_change_idx = a_cat[no_change]   
        

        if idx in ordinal:
            neighbourhood[change_idx, idx] = change_ordinal_vars(neighbourhood[change_idx, idx], idx, categorical_names)
        else:
            neighbourhood[change_idx, idx] = change_categorical_vars(neighbourhood[change_idx, idx], idx, categorical_names)

        neighbourhood[no_change_idx, idx] = sampled_nn[no_change_idx, idx]
    
    return neighbourhood


def medoid_neighbourhood(point, idx, labels, num, knn_overall, centers, alpha, alpha_cat, categorical_features, discrete, ordinal, categorical_names, num_ft_or):
    x = np.full(shape = (num+1, num_ft_or), fill_value=point)

    cluster_id = int(labels[idx])
    knn_points = knn_overall[cluster_id]

    x[1:, :] = medoid_perturbation(x[1:,:], knn_points, centers, alpha, alpha_cat, categorical_features, discrete, ordinal, categorical_names, num_ft_or)

    return x