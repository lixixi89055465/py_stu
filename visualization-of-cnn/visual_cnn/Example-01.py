import torch
from torch import nn
import torchvision

print(torch.__version__)
# model1 = torch.load(r'')

model1 = torchvision.models.resnet18(pretrained=True)
model_weight = []
conv_layers = []
counter = 0
model_children = list(model1.children())
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weight.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    model_weight.append(child.weight)
                    conv_layers.append(child)
print(f'Total convolution layers :{counter}')
print('conv_layers')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mode1 = model1.to(device)
print(model1.children())
# print(list(model1.children()))

from torchvision import transforms

training_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
validation_transforms = transforms.Compose([
    transforms.RandomResizedCrop(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
testing_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

from PIL import Image

image = Image.open(str(r'./a.jpg'))
image = testing_transforms(image)
print(f'Image shape before:{image.shape}')
image = image.unsqueeze(0)
print(f'Image shape after:{image.shape}')
image = image.to(device)
outputs = []
names = []
for layer in conv_layers[0:]:
    image = layer(image)
    outputs.append(image)
    names.append(str(layer))
# print(outputs)
print(outputs[1].shape)

print(torch.sum(outputs[1].squeeze(0), 0).shape)

processed = []
for feature_map in outputs:
    feature_map = feature_map.squeeze(0)
    gray_scale = feature_map.sum(0)
    gray_scale = gray_scale / feature_map.shape[0]
    processed.append(gray_scale.data.cpu().numpy())

for fm in processed:
    print(fm.shape)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(30, 50))
for i in range(len(processed)):
    a = fig.add_subplot(5, 4, i + 1)
    imgplot = plt.imshow(processed[i])
    a.axis('off')
    a.set_title(names[i].split('(')[0], fontsize=30)
plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')
