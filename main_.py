# -*- coding: utf-8 -*-


import os
from sklearn.preprocessing import StandardScaler
import timm
import matplotlib.pyplot as plt
from pyts.datasets import load_gunpoint
from pyts.image import RecurrencePlot
from torch.utils.data import Dataset, DataLoader
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
from pyts.image import MarkovTransitionField,RecurrencePlot
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import warnings
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from aeon.datasets import load_classification
import sklearn
from sklearn.metrics import pairwise_kernels
from sklearn import metrics
from sklearn.cluster import KMeans,SpectralClustering,AgglomerativeClustering
from sklearn.decomposition import PCA,KernelPCA
from sklearn.mixture import GaussianMixture
import pandas as pd  # requires: pip install pandas
import torch
from chronos import BaseChronosPipeline
from data import *
import os
import numpy as np
from momentfm import MOMENTPipeline
import umap
warnings.filterwarnings("ignore")
random.seed(2025)  
np.random.seed(2025)

pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
    device_map="cuda",  # use "cpu" for CPU inference
    torch_dtype=torch.bfloat16,
)



#X = torch.randn(16, 1, 512)


#output = model(x_enc=X)



validation_datasets = os.listdir('UCR2018')

clustering_method_ = ['k_means','gaussian','Agglomerative']
reduction_method_ = ['pca','umap']

for clustering_method in clustering_method_:
    for reduction_method in reduction_method_:
        file_name = 'All128datasets_' + 'Chronos_' + clustering_method + '_' + reduction_method + '_' + '.csv'
        
        for dataset in validation_datasets:
            save_results = {'Dataset': list(),
                                'ARI': list(),
                                'NMI': list()}
            print(dataset)
            save_results['Dataset'].append(dataset)
            X, Y = get_UCR_datasets(dataset)
            num_clusters = len(np.unique(Y))
            X_features = np.zeros([len(X),512])   
            for i in range(len(X)):
                temp_x = torch.tensor(X[i])
                embeddings, tokenizer_state = pipeline.embed(temp_x)
                mean_vector = embeddings.mean(dim=1)  # [1, 512]
                mean_vector = mean_vector.squeeze(0)  # [512]
                X_features[i,:] = mean_vector.float()
                
                
            if reduction_method == 'pca':
                pca = PCA().fit(X_features)
                optimal_dimensions = np.argmax(pca.explained_variance_ratio_ < 0.01)
                pca_optimal = PCA(n_components=optimal_dimensions)
                transformed_data_pca = pca_optimal.fit_transform(X_features)
            else:
                reducer = umap.UMAP(n_components=10,random_state=42)
                transformed_data_pca = reducer.fit_transform(X_features)
                
            if clustering_method=='k_means':
                labels_pred = KMeans(n_clusters=num_clusters, n_init=10).fit_predict(transformed_data_pca)
                score = metrics.adjusted_rand_score(labels_true=Y, labels_pred=labels_pred)
                save_results['ARI'].append(score)
                score = metrics.normalized_mutual_info_score(labels_true=Y, labels_pred=labels_pred)
                save_results['NMI'].append(score)
            elif clustering_method=='gaussian':
                labels_pred = GaussianMixture(n_components=num_clusters).fit_predict(transformed_data_pca)
                score = metrics.adjusted_rand_score(labels_true=Y, labels_pred=labels_pred)
                save_results['ARI'].append(score)
                score = metrics.normalized_mutual_info_score(labels_true=Y, labels_pred=labels_pred)
                save_results['NMI'].append(score)
            else:
                labels_pred = AgglomerativeClustering(n_clusters=num_clusters).fit_predict(transformed_data_pca)
                score = metrics.adjusted_rand_score(labels_true=Y, labels_pred=labels_pred)
                save_results['ARI'].append(score)
                score = metrics.normalized_mutual_info_score(labels_true=Y, labels_pred=labels_pred)
                save_results['NMI'].append(score)
                
            df = pd.DataFrame(save_results)
            df.to_csv(file_name,mode='a', index=False, header=False)




