# Import necessary packages.
import numpy as np
import torch
import torch.nn as nn
# import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder

# This is for the progress bar.
from tqdm.auto import tqdm

# It is important to do images augmentation in training.
# However, not every augmentation is useful.
# Please think about what kind of augmentation is helpful for food recognition.
# train_tfm = transforms.Compose([
#     # Resize the image into a fixed shape (height = width = 128)
#     transforms.Resize((128, 128)),
#     # You may add some transforms here.
#     # ToTensor() should be the last one of the transforms.
#     transforms.ToTensor(),
# ])

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
# test_tfm = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
# ])

# Batch size for training, validation, and testing.
# A greater batch size usually gives a more stable gradient.
# But the GPU memory is limited, so please adjust it carefully.
batch_size = 128

# Construct datasets.
# The argument "loader" tells how torchvision reads the images.
train_set = DatasetFolder("../images/food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg")

valid_set = DatasetFolder("../images/food-11/validation", loader=lambda x: Image.open(x), extensions="jpg")
unlabeled_set = DatasetFolder("../images/food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg")
test_set = DatasetFolder("../images/food-11/testing", loader=lambda x: Image.open(x), extensions="jpg")


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x = self.cnn_layers(x)

        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x


def get_pseudo_labels(dataset, model, threshold=0.01):
    # This functions generates pseudo-labels of a dataset using given model.
    # It returns an instance of DatasetFolder containing images whose prediction confidences exceed a given threshold.
    # You are NOT allowed to use any models trained on external images for pseudo-labeling.
    # print(unlabeled_set.samples)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Make sure the model is in eval mode.
    # Define softmax function.
    # softmax = nn.Softmax(dim=-1)
    softmax = nn.Softmax(dim=0)
    labels = torch.tensor([], device=device)
    # Iterate over the dataset by batches.
    # for batch in tqdm(dataloader):
    targets = []
    for batch in tqdm(dataset):
        img, _ = batch

        # Forward the images
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits =torch.rand(128,11)

            # Obtain the probability distributions by applying softmax on logits.
            probs = softmax(logits)

            # ---------- TODO ----------
            # Filter the images and construct a new dataset.
            # probs = probs[probs.max(dim=1) > threshold]
            # print(probs.max(dim=1))
            # print(probs.max(dim=1)[0] > threshold)
            # print(probs[probs.max(dim=1)[0] > threshold])
            targets += [probs.max(dim=1)[0] > threshold]
            # if probs.shape[0] > 0:
                # print(probs)
                # img_label = torch.argmax(probs, dim=1)
                # labels = torch.cat([labels, img_label], dim=0).to(device)

            # print(torch.argmax(probs, dim=1))

    # # Turn off the eval mode.
    # model.train()
    print('targets.len:',len(targets))
    dataset.samples = dataset.samples[targets]
    dataset.targets = dataset.targets[targets]
    return dataset


model = Classifier()
pseudo_set = get_pseudo_labels(unlabeled_set, model=model)

concat_dataset = ConcatDataset([train_set, pseudo_set])
print(concat_dataset)
train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
