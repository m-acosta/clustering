#-------------------------------------------------------------------------
# AUTHOR: Michael Acosta
# FILENAME: clustering.py
# SPECIFICATION: This program runs k-means with k values from 2 to 20 to 
#                   find the best value. The silhouette coefficients are 
#                   plotted and the Homogeneity Score is shown for the 
#                   best k value.
# FOR: CS 4210- Assignment #5
# TIME SPENT: ~1 hour
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = np.array(df.values)[:,:64]

#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code

silhouette = {}
max_score = 0
best_k = 0

for k in range(2, 21):
     kmeans = KMeans(n_clusters=k, random_state=0)
     kmeans.fit(X_training)


     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     silhouette[k] = silhouette_score(X_training, kmeans.labels_)
     if silhouette[k] > max_score:
          max_score = silhouette[k]
          best_k = k

#use the k value that maximizes the silhouette coefficient
print("The best k value is: " + str(best_k))
kmeans = KMeans(n_clusters=best_k, random_state=0, n_init=10)
kmeans.fit(X_training)

# plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
plt.plot(silhouette.keys(), silhouette.values())
plt.show()

#reading the test data (clusters) by using Pandas library
df_test = pd.read_csv('testing_data.csv', sep=',', header=None)

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
labels = np.array(df_test.values).reshape(1,len(df_test))[0]

#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
