#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df2 = pd.read_csv("animelists_cleaned.csv")
df2_v2 = df2[['username','anime_id','my_watched_episodes','my_score','my_status']]
df2_v2


# In[3]:


df3 = pd.read_csv("anime_cleaned.csv")
df3_v2 = df3[["anime_id",'title','genre','source','popularity','rank','type','episodes','favorites']]
df3_v2


# In[4]:


df3.info()


# In[5]:


df4 = pd.read_csv("UserList.csv")
df4_v2 = df4[['username','user_id']]


# In[6]:


merged_df = pd.merge(df2_v2 , df3_v2, on='anime_id', how = 'right')


# In[7]:


merged_df


# In[8]:


merged_df2 = pd.merge(merged_df, df4_v2, on = "username", how = "right")
merged_df2


# In[9]:


merged_df2.isnull().sum()


# In[10]:


merged_df2 = merged_df2.dropna()
merged_df2.isnull().sum()


# In[11]:


merged_df2


# In[12]:


merged_df2.corr()


# high correlation
# my_watched_episodes and episodes
# 
# popularity and favourites
# 
# popularity and rank
# 

# In[13]:


small_df = merged_df2.sample(frac = 0.0001) #My macbook can't handle 30Million hahaha so I've cut down but feel free to change back when using desktop
small_df


# In[14]:


small_df['source'].value_counts()


# In[15]:


small_df2 = small_df.drop(['username'], axis = 1)


# In[16]:


small_df2


# In[17]:


small_df2['type'].value_counts()


# In[18]:


small_df3 = small_df2.drop('title', axis=1)


# In[19]:


small_df3


# In[20]:


# Create a list of unique genre labels
genres = set([genre for row in small_df3['genre'].str.split(', ') for genre in row])

# One Hot Encoding
for genre in genres:
    small_df3[genre] = small_df3['genre'].apply(lambda x: 1 if genre in x.split(', ') else 0)

# Drop the original 'genre' column
small_df3.drop('genre', axis=1, inplace=True)

small_df4 = pd.concat([small_df3.iloc[:,:11], small_df3.iloc[:, -len(genres):]], axis=1)
small_df4


# In[21]:


small_df4


# In[22]:


sns.scatterplot(data = small_df4, x = 'my_score', y ='episodes')


# In[23]:


sns.scatterplot(data = small_df4, x = 'my_watched_episodes', y='episodes')


# In[24]:


small_df4[['type']].value_counts()


# In[25]:


import pandas as pd

#One Hot Encoding
type_dummies = pd.get_dummies(small_df4['type'])

small_df5 = pd.concat([small_df4, type_dummies], axis=1)

# Drop the original 'genre' column
small_df5.drop('type', axis=1, inplace=True)
small_df5


# In[26]:


small_df4[['source']].value_counts()


# In[27]:


import gensim

# Using Word2Vec on the source column
sources = [source.split(", ") for source in small_df5["source"]]

model = gensim.models.Word2Vec(sources, vector_size=100, min_count=1)


source_dict = {}
for source in set(small_df5["source"]):
    source_dict[source] = model.wv[source.split(", ")[0]]

# Create a new dataframe 
source_df = small_df4[["anime_id", "source"]].copy()
source_df["source_vector"] = source_df["source"].apply(lambda x: model.wv[x.split(", ")[0]])
source_df = pd.concat([source_df[["anime_id", "source_vector"]], source_df["source_vector"].apply(pd.Series)], axis=1)

anime_df = pd.merge(small_df5.drop(columns=["source"]), source_df, on="anime_id")


# In[28]:


anime_df


# In[29]:


anime_df['user_id']


# In[30]:


anime_df.isnull().any()


# In[31]:


anime_df.corr()


# In[32]:


anime_df2 = anime_df.drop(['my_status', 'episodes', 'rank', 'popularity'], axis =1)
anime_df2


# In[33]:


from sklearn.preprocessing import LabelEncoder

# Label Encoding
le = LabelEncoder()


for col in anime_df2.select_dtypes(include='object').columns:
    anime_df2[col] = le.fit_transform(anime_df2[col].astype(str))


# In[34]:


anime_df2


# In[35]:


print(anime_df2.dtypes)


# In[36]:


svd_df = anime_df2.drop(['user_id'], axis =1 )


# In[37]:


X = svd_df.values
mean = np.mean(X, axis = 0)
X_prime = X - mean


# In[38]:


X_prime.shape


# In[39]:


X_prime = X_prime.astype(float)


# In[40]:


from scipy.linalg import svd
U,D,VT = svd(X_prime)


# In[41]:


U.shape, D.shape, VT.shape


# In[42]:


# Assuming q = 3
q = 3

U_q = U[:, :q]
D_q = D[:q]
VT_q = VT[:q, :]

# Compute the low-rank truncation of X_prime
X_prime_q = U_q @ np.diag(D_q) @ VT_q
X_prime_q


# In[43]:


original_data = X_prime
reconstruct_data = X_prime_q

MSE = np.mean(np.sqrt((original_data - reconstruct_data)**2))

print("Error: ", MSE)


# The Error looks promising, for now

# In[44]:


import numpy as np
from sklearn.cluster import KMeans

# perform k-means clustering on the low-rank approximation, 
#we could also do it manually I've posted the code on the last part
kmeans = KMeans(n_clusters=2, random_state=0).fit(reconstruct_data)

labels = kmeans.labels_


# In[45]:


from sklearn.metrics import silhouette_score

y_pred = labels

# Compute the silhouette score
score = silhouette_score(reconstruct_data, y_pred)
print("Silhouette score: ", score.round(2))


# In[46]:


X


# ## Option 1: Pearson Correlation

# In[47]:


import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

# Compute the item-item similarity matrix using Pearson correlation
item_sim_matrix = np.corrcoef(X_prime_q.T)


weighted_ratings = item_sim_matrix.dot(X_prime_q.T)

weights = np.abs(item_sim_matrix).sum(axis=1)
weighted_ratings /= weights.reshape(-1, 1)

user_id = 10 # assuming the user ID is 10
recommended_items = np.argsort(weighted_ratings[:, user_id])[::-1][:10]


# In[48]:


recommended_items


# In[49]:



recommended_df = pd.DataFrame({'item_id': recommended_items, 'score': weighted_ratings[:, user_id][recommended_items]})

# Sort the DataFrame by score in descending order to get the top recommendations
recommended_df = recommended_df.sort_values(by='score', ascending=False)

final_df = pd.merge(recommended_df, small_df, how='inner', left_on='item_id', right_on='anime_id')
final_df.head(10)


# ## Option 1: Cosine Similarity

# In[50]:


from sklearn.metrics.pairwise import cosine_similarity
# Compute the item-item similarity matrix
item_sim_matrix = np.corrcoef(X_prime_q.T)

user_ratings = reconstruct_data[:, ].reshape(-1, 1)

weights = np.abs(item_sim_matrix).sum(axis=1)
weighted_ratings /= weights.reshape(-1, 1)

recommended_items = np.argsort(weighted_ratings.flatten())[::-1][:10]


# In[51]:


recommended_items


# ## K Nearest Neighbour

# In[52]:


from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

# Compute the item-item similarity matrix
item_sim_matrix = cosine_similarity(reconstruct_data.T)

# Create a NearestNeighbors model using the item-item similarity matrix
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(reconstruct_data.T)

k = 10 # number of nearest neighbors to consider
item_indices = model_knn.kneighbors(return_distance=False, n_neighbors=k)

weights = item_sim_matrix.sum(axis=1).reshape(-1, 1)
weighted_ratings = item_sim_matrix.T.dot(reconstruct_data.T) / weights

user_id = 300  # assuming the user ID is 300
recommended_items = np.argsort(weighted_ratings[:, user_id])[::-1][:10]


# In[53]:


recommended_items


# In[54]:



recommended_df = pd.DataFrame({'item_id': recommended_items, 'score': weighted_ratings[:, user_id][recommended_items]})

# Sort the DataFrame by score in descending order to get the top recommendations
recommended_df = recommended_df.sort_values(by='score', ascending=False)
final_df = pd.merge(recommended_df, small_df, how='inner', left_on='item_id', right_on='anime_id')

final_df.head(10)


# In[55]:


import pandas as pd

def get_recommendations(data_list):
    anime_df = pd.read_csv('animelist.csv')

    merged_df = pd.merge(data_list, anime_df, on='anime_id')
    
    columns_to_keep = ['anime_id', 'Name', 'Score', 'Type', 'Source', 'Synopsis']
    recommended_anime = merged_df[columns_to_keep]
    
    recommended_anime = recommended_anime.iloc[recommended_items][:10]

    return recommended_anime


# In[56]:


recommended_df.to_csv('recommended_anime.csv', index=False)


# In[57]:


#recommended_df = recommended_df(data)


# In[1]:


#Complete Code
# def euclidean_dissimilarity(a, b):
#     return np.linalg.norm(a - b)

# def random_vectors(Matrix_X, K):
#     return Matrix_X[np.random.choice(Matrix_X.shape[0], K, replace=False), :]

# def assign_cluster(x_i, dissimilarity, mean_vectors):
#     min_dissimilarity = float("inf")
#     best_cluster = -2
    
#     for k, mean_vector in enumerate(mean_vectors):
#         cur_dissimilarity = dissimilarity(x_i, mean_vector)
#         if cur_dissimilarity < min_dissimilarity:
#             min_dissimilarity = cur_dissimilarity
#             best_cluster = k
#     return best_cluster

# def k_means(X_Matrix, K, dissimilarity, mean_vectors=None, max_iters=500):
#     N = X_Matrix.shape[0]
#     if mean_vectors is None:
#         mean_vectors = random_vectors(X_Matrix, K)
#     labels = np.zeros(N)
    
#     for i in range(max_iters):
#         for j in range(N):
#             labels[j] = assign_cluster(X_Matrix[j], dissimilarity, mean_vectors)
            
#         for k in range(K):
#             mean_vectors[k] = np.mean(X_Matrix[labels == k], axis=0)
            
#     return labels, mean_vectors

# K = 2
# labels, means = k_means(projected_data, K, euclidean_dissimilarity)


# In[ ]:




