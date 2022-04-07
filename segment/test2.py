'''

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from skimage.color import rgb2gray,rgb2hsv,hsv2rgb
from skimage.io import imread,imshow
from sklearn.cluster import KMeans


dog = imread('beach_doggo.PNG')
plt.figure(num=None, figsize=(8, 6), dpi=80)
imshow(dog)
plt.show()

def image_to_pandas(image):
    df = pd.DataFrame([image[:,:,0].flatten(),
                       image[:,:,1].flatten(),
                       image[:,:,2].flatten()]).T
    df.columns = ['Red_Channel','Green_Channel','Blue_Channel']
    return df
df_doggo = image_to_pandas(dog)
df_doggo.head(5)


plt.figure(num=None, figsize=(8, 6), dpi=80)
kmeans = KMeans(n_clusters=  4, random_state = 42).fit(df_doggo)
result = kmeans.labels_.reshape(dog.shape[0],dog.shape[1])
imshow(result, cmap='viridis')
plt.show()