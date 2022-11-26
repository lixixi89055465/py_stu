import torch
from torch import nn
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.__version__)
# model1 = torch.load(r'')

# model1 = torchvision.models.resnet18(pretrained=True)
model = torchvision.models.alexnet()
mode1 = model.to(device)
model.eval()
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

img_path = './a.jpg'
from PIL import Image

import matplotlib.pyplot as plt

img = Image.open(img_path)
img = img.convert('RGB')
img = transform(img).to(device)
img = img.unsqueeze(0)
print(model.features[0])
import numpy as np


# with torch.no_grad():
#     output = model.features[0](img)
#     for i in range(output.shape[1]):
#         img = output[:, i, :, :].numpy()
#         min = np.min(img)
#         max = np.max(img)
#         img = (img - min) / (max - min)
#         img = img * 255
#         im = Image.fromarray(img[0])
#         im = im.convert('L')

def saveFig(img, output, index):
    with torch.no_grad():
        fig = plt.figure(figsize=(30, 50))
        for i in range(output.shape[1]):
            img = output[:, i, :, :].squeeze(0).cpu().numpy()
            min = np.min(img)
            max = np.max(img)
            img = (img - min) / (max - min)
            img = img * 255
            a = fig.add_subplot(8, output.shape[1] // 8, i + 1)
            imgplt = plt.imshow(img)
            a.axis('off')
        plt.savefig("FeatureMap2/feature_maps" + str(index) + ".jpg", bbox_inches='tight')


featureCNN = model.features
index = 1
for child in featureCNN:
    img = child(img)
    if type(child) == nn.Conv2d:
        saveFig(child, output=img, index=index)
    index += 1
