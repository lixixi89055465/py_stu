import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans



img = cv2.imread(r'1.png')#BGR format
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
raw_img = img.copy()

print(img.shape)
width = img.shape[0]
height = img.shape[1]

#we are creating segmentation of K colors
#firstly, we will find K dominating colors in an image using K means clustering

#we have RGB image(3 channels) so we will use 3 channels values as features
img = img.reshape(-1,3)
print(img.shape)

k = 4   #number of dominating colors you want

classifier = KMeans(n_clusters=k,max_iter=100,
                init='k-means++')
kmeans = classifier.fit(img)

labels = kmeans.labels_
print(f'labels:{labels}')

dominating_colors = np.array(kmeans.cluster_centers_,dtype='uint8')

print(dominating_colors)
colors = []

for col in dominating_colors:
	r = col[0]
	g = col[1]
	b = col[2]
	colors.append([r, g, b])

segmented_img = np.zeros((img.shape[0],3),dtype='uint8')

for pix in range(segmented_img.shape[0]):
	r_g_b = colors[labels[pix]]
	segmented_img[pix] = r_g_b

segmented_img = segmented_img.reshape((width,height,3))
plt.subplot(121)
plt.imshow(raw_img)
plt.subplot(122)
plt.imshow(segmented_img)
plt.show()
