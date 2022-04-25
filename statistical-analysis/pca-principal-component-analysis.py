

# PCA - principal component analysis
# start by Standardizing the data since PCA's output is influenced based on the scale of the features of the data
from sklearn.preprocessing import StandardScaler
x = breast_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x) # normalizing the features

# check whether the normalized data has a mean of zero and a standard deviation of one
np.mean(x),np.std(x)

# convert the normalized features into a tabular format with the help of DataFrame
feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised_breast = pd.DataFrame(x,columns=feat_cols)
normalised_breast.tail()

# projecting the multi-dimensional data to two-dimensional principal components
from sklearn.decomposition import PCA
pca_breast = PCA(n_components=2)
principalComponents_breast = pca_breast.fit_transform(x)

# create a DataFrame that will have the principal component values for all 569 samples
principal_breast_Df = pd.DataFrame(data = principalComponents_breast
             , columns = ['principal component 1', 'principal component 2'])
principal_breast_Df.tail()

# find the explained_variance_ratio. It will provide you with the amount of information or variance each principal component holds after projecting the data to a lower dimensional subspace
print('Explained variation per principal component: {}'.format(pca_breast.explained_variance_ratio_))
# Example - if you observe that the principal component 1 holds 44.2% 
# of the information while the principal component 2 holds only 19% of 
# the information, then 36.8% information was lost while projecting 
# multi-dimensional data to a two-dimensional data


