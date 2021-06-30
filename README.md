
# K- MEANS
K-means clustering is a type of unsupervised learning, which is used when you have unlabeled data (i.e., data without defined categories or groups). The goal of this algorithm is to find groups in the data, with the number of groups represented by the variable K. The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity. The results of the K-means clustering algorithm are:

The centroids of the K clusters, which can be used to label new data

Labels for the training data (each data point is assigned to a single cluster)

Rather than defining groups before looking at the data, clustering allows you to find and analyze the groups that have formed organically. The "Choosing K" section below describes how the number of groups can be determined.  

Each centroid of a cluster is a collection of feature values which define the resulting groups. Examining the centroid feature weights can be used to qualitatively interpret what kind of group each cluster represents.  

The Κ-means clustering algorithm uses iterative refinement to produce a final result. The algorithm inputs are the number of clusters Κ and the data set. The data set is a collection of features for each data point. The algorithms starts with initial estimates for the Κ centroids, which can either be randomly generated or randomly selected from the data set. The algorithm then iterates between two steps:

1. Data assigment step:

Each centroid defines one of the clusters. In this step, each data point is assigned to its nearest centroid, based on the squared Euclidean distance. More formally, if ci is the collection of centroids in set C, then each data point x is assigned to a cluster based on

$$\underset{c_i \in C}{\arg\min} \; dist(c_i,x)^2$$

where dist( · ) is the standard (L2) Euclidean distance. Let the set of data point assignments for each ith cluster centroid be Si.

2. Centroid update step:

In this step, the centroids are recomputed. This is done by taking the mean of all data points assigned to that centroid's cluster.
$$c_i=\frac{1}{|S_i|}\sum_{x_i \in S_i x_i}$$



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

```


```python
df = pd.read_csv('vinos.csv')
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 178 entries, 0 to 177
    Data columns (total 14 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   Vino             178 non-null    int64  
     1   Alcohol          178 non-null    float64
     2   Malic            178 non-null    float64
     3   Ash              178 non-null    float64
     4   Alcalinity       178 non-null    float64
     5   Magnesium        178 non-null    int64  
     6   Phenols          178 non-null    float64
     7   Flavanoids       178 non-null    float64
     8   Nonflavanoids    178 non-null    float64
     9   Proanthocyanins  178 non-null    float64
     10  Color            178 non-null    float64
     11  Hue              178 non-null    float64
     12  Dilution         178 non-null    float64
     13  Proline          178 non-null    int64  
    dtypes: float64(11), int64(3)
    memory usage: 19.6 KB



```python
df.head()
df.drop(['Vino'], axis=1, inplace=True)
```


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Alcohol</th>
      <th>Malic</th>
      <th>Ash</th>
      <th>Alcalinity</th>
      <th>Magnesium</th>
      <th>Phenols</th>
      <th>Flavanoids</th>
      <th>Nonflavanoids</th>
      <th>Proanthocyanins</th>
      <th>Color</th>
      <th>Hue</th>
      <th>Dilution</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>13.000618</td>
      <td>2.336348</td>
      <td>2.366517</td>
      <td>19.494944</td>
      <td>99.741573</td>
      <td>2.295112</td>
      <td>2.029270</td>
      <td>0.361854</td>
      <td>1.590899</td>
      <td>5.058090</td>
      <td>0.957449</td>
      <td>2.611685</td>
      <td>746.893258</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.811827</td>
      <td>1.117146</td>
      <td>0.274344</td>
      <td>3.339564</td>
      <td>14.282484</td>
      <td>0.625851</td>
      <td>0.998859</td>
      <td>0.124453</td>
      <td>0.572359</td>
      <td>2.318286</td>
      <td>0.228572</td>
      <td>0.709990</td>
      <td>314.907474</td>
    </tr>
    <tr>
      <th>min</th>
      <td>11.030000</td>
      <td>0.740000</td>
      <td>1.360000</td>
      <td>10.600000</td>
      <td>70.000000</td>
      <td>0.980000</td>
      <td>0.340000</td>
      <td>0.130000</td>
      <td>0.410000</td>
      <td>1.280000</td>
      <td>0.480000</td>
      <td>1.270000</td>
      <td>278.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>12.362500</td>
      <td>1.602500</td>
      <td>2.210000</td>
      <td>17.200000</td>
      <td>88.000000</td>
      <td>1.742500</td>
      <td>1.205000</td>
      <td>0.270000</td>
      <td>1.250000</td>
      <td>3.220000</td>
      <td>0.782500</td>
      <td>1.937500</td>
      <td>500.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>13.050000</td>
      <td>1.865000</td>
      <td>2.360000</td>
      <td>19.500000</td>
      <td>98.000000</td>
      <td>2.355000</td>
      <td>2.135000</td>
      <td>0.340000</td>
      <td>1.555000</td>
      <td>4.690000</td>
      <td>0.965000</td>
      <td>2.780000</td>
      <td>673.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>13.677500</td>
      <td>3.082500</td>
      <td>2.557500</td>
      <td>21.500000</td>
      <td>107.000000</td>
      <td>2.800000</td>
      <td>2.875000</td>
      <td>0.437500</td>
      <td>1.950000</td>
      <td>6.200000</td>
      <td>1.120000</td>
      <td>3.170000</td>
      <td>985.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>14.830000</td>
      <td>5.800000</td>
      <td>3.230000</td>
      <td>30.000000</td>
      <td>162.000000</td>
      <td>3.880000</td>
      <td>5.080000</td>
      <td>0.660000</td>
      <td>3.580000</td>
      <td>13.000000</td>
      <td>1.710000</td>
      <td>4.000000</td>
      <td>1680.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Normalizar valores


```python
df_norm = (df - df.min())/(df.max()-df.min())
df_norm.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Alcohol</th>
      <th>Malic</th>
      <th>Ash</th>
      <th>Alcalinity</th>
      <th>Magnesium</th>
      <th>Phenols</th>
      <th>Flavanoids</th>
      <th>Nonflavanoids</th>
      <th>Proanthocyanins</th>
      <th>Color</th>
      <th>Hue</th>
      <th>Dilution</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.842105</td>
      <td>0.191700</td>
      <td>0.572193</td>
      <td>0.257732</td>
      <td>0.619565</td>
      <td>0.627586</td>
      <td>0.573840</td>
      <td>0.283019</td>
      <td>0.593060</td>
      <td>0.372014</td>
      <td>0.455285</td>
      <td>0.970696</td>
      <td>0.561341</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.571053</td>
      <td>0.205534</td>
      <td>0.417112</td>
      <td>0.030928</td>
      <td>0.326087</td>
      <td>0.575862</td>
      <td>0.510549</td>
      <td>0.245283</td>
      <td>0.274448</td>
      <td>0.264505</td>
      <td>0.463415</td>
      <td>0.780220</td>
      <td>0.550642</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.560526</td>
      <td>0.320158</td>
      <td>0.700535</td>
      <td>0.412371</td>
      <td>0.336957</td>
      <td>0.627586</td>
      <td>0.611814</td>
      <td>0.320755</td>
      <td>0.757098</td>
      <td>0.375427</td>
      <td>0.447154</td>
      <td>0.695971</td>
      <td>0.646933</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.878947</td>
      <td>0.239130</td>
      <td>0.609626</td>
      <td>0.319588</td>
      <td>0.467391</td>
      <td>0.989655</td>
      <td>0.664557</td>
      <td>0.207547</td>
      <td>0.558360</td>
      <td>0.556314</td>
      <td>0.308943</td>
      <td>0.798535</td>
      <td>0.857347</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.581579</td>
      <td>0.365613</td>
      <td>0.807487</td>
      <td>0.536082</td>
      <td>0.521739</td>
      <td>0.627586</td>
      <td>0.495781</td>
      <td>0.490566</td>
      <td>0.444795</td>
      <td>0.259386</td>
      <td>0.455285</td>
      <td>0.608059</td>
      <td>0.325963</td>
    </tr>
  </tbody>
</table>
</div>



## Cantidad de K
Codo de Jambu: Usar diferentes cluster e ir viendo que tan similares son los individuos dentro de los clusters


```python
similaridad = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, max_iter = 300)
    kmeans.fit(df_norm)
    similaridad.append(kmeans.inertia_)
```


```python
plt.plot(range(1,11), k)
plt.title('Codo de Jambu')
plt.xlabel('Numero de CLusters')
plt.ylabel('Similaridad')
```




    Text(0, 0.5, 'Similaridad')




![png](output_10_1.png)


## Aplicando K-Means


```python
clustering = KMeans(n_clusters=3, max_iter=300)
clustering.fit(df_norm)

```




    KMeans(n_clusters=3)



## Agregando al archivo original


```python
df['Cluster'] = clustering.labels_
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Alcohol</th>
      <th>Malic</th>
      <th>Ash</th>
      <th>Alcalinity</th>
      <th>Magnesium</th>
      <th>Phenols</th>
      <th>Flavanoids</th>
      <th>Nonflavanoids</th>
      <th>Proanthocyanins</th>
      <th>Color</th>
      <th>Hue</th>
      <th>Dilution</th>
      <th>Proline</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Visualizando


```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_vinos = pca.fit_transform(df_norm)
pca_vinos_df = pd.DataFrame(data=pca_vinos, columns=['Componente_1', 'Componente_2'])
pca_nombres_vinos = pd.concat([pca_vinos_df, df[['Cluster']]], axis=1)
```


```python
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Componente_1', fontsize=14)
ax.set_ylabel('Componente_2', fontsize=14)
ax.set_title('Componentes principales', fontsize=20)

colores = np.array(['blue','red','orange'])
ax.scatter(x = pca_nombres_vinos.Componente_1, y=pca_nombres_vinos.Componente_2, c=colores[pca_nombres_vinos.Cluster], s=50)
```




    <matplotlib.collections.PathCollection at 0x7f81cf4831d0>




![png](output_17_1.png)



```python
df.to_csv('vinos_kmeans.csv')
```
