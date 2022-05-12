
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
#loading the provided data
stock_data = np.loadtxt("stocks.csv", delimiter = ",", skiprows = 1)

#splitting the data into domain and range values
X = stock_data[:,0:3]

#Scale the data, 0 - 1 MinMaxScaler
X = MinMaxScaler().fit_transform(X)

#print(X)
#Identify stock tickers
tick = stock_data[:,4:5]

#DBSCAN 
db = DBSCAN(eps = 0.25, min_samples = 2).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

#labels
labels = db.labels_
print("The clusters assigned to respective stock tickers [1-30]: " +  "\n" + str(labels))


# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)





