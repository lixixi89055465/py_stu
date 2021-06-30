import torch
import matplotlib.pyplot as plt


# import torch.nn as nn
# import torch.nn.functional as F

def show_tensor_images(image_tensor, num_images=16, size=(3, 64, 64), nrow=3):
    '''
    Function for visualizing images: Given a tensor of images, number of images,
    size per image, and images per row, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu().clamp_(0, 1)
    image_grid = torch.nn.functional.make_grid(image_unflat[:num_images], nrow=nrow, padding=0)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')
    plt.show()


from scipy.stats import truncnorm


def get_truncated_noise(n_sample, z_dim, truncation):
    truncated_noise = truncnorm.rvs(-1 * truncation, truncation, size=(n_sample, z_dim))
    return torch.Tensor(truncated_noise)


assert tuple(get_truncated_noise(n_sample=10, z_dim=5, truncation=0.7).shape) == (10, 5)
simple_noise = get_truncated_noise(n_sample=1000, z_dim=10, truncation=0.2)
assert simple_noise.max() > 0.199 and simple_noise.max() < 2
assert simple_noise.min() < -0.199 and simple_noise.min() > -0.2
assert simple_noise.std() > 0.113 and simple_noise.std() < 0.117

print("Success!")


class MappingLayers(torch.nn.Module):
    def __init__(self, z_dim, hidden_dim, w_dim):
        super().__init__()
        self.mapping = torch.nn.Sequential(
            torch.nn.Linear(z_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, w_dim)
        )

    def forward(self, noise):
        return self.mapping(noise)

    def get_mapping(self):
        return self.mapping


# Test  the mapping function
map_fn = MappingLayers(10, 20, 30)
a = map_fn(torch.randn(2, 10))
print(a.shape)
assert tuple(a.shape) == (2, 30)
# assert tuple(map_fn(torch.randn(2, 10)).shape) == (2, 30)
assert len(map_fn.mapping) > 4
outputs = map_fn(torch.randn(1000, 10))
assert outputs.std() > 0.05 and outputs.std() < 0.3
assert outputs.min() > -2 and outputs.min() < 0
assert outputs.max() < 2 and outputs.max() > 0
layers = [str(x).replace(' ', '').replace('inplace=True', '') for x in map_fn.get_mapping()]
print(map_fn.get_mapping())

assert layers == ['Linear(in_features=10,out_features=20,bias=True)',
                  'ReLU()',
                  'Linear(in_features=20,out_features=20,bias=True)',
                  'ReLU()',
                  'Linear(in_features=20,out_features=30,bias=True)']
print("Success!")


class InjectNoise(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(channels)[None, :, None, None]
        )

    def forward(self, image):
        noise_shape = (image.shape[0], 1, image.shape[2], image.shape[3])
        noise = torch.randn(noise_shape, device=image.device)
        # return image + self.weight * noise
        a = self.weight * noise
        return image + a

    def get_weight(self):
        return self.weight

    def get_self(self):
        return self


test_noise_channels = 3000
test_noise_samples = 20
fake_images = torch.randn(test_noise_samples, test_noise_channels, 10, 10)
inject_noise = InjectNoise(test_noise_channels)
print(inject_noise.weight.std())
print(inject_noise.weight.shape)
assert torch.abs(inject_noise.weight.std() - 1) < 0.1
assert torch.abs(inject_noise.weight.mean()) < 0.1
assert type(inject_noise.get_weight() == torch.nn.parameter.Parameter)
###
assert tuple(inject_noise.weight.shape) == (1, test_noise_channels, 1, 1)
inject_noise.weight = torch.nn.Parameter(torch.ones_like(inject_noise.weight))
# Check that something changed
assert torch.abs((inject_noise(fake_images) - fake_images)).mean() > 0.1
# Check that the change is per-channel
assert torch.abs((inject_noise(fake_images) - fake_images).std(0)).mean() > 1e-4
assert torch.abs((inject_noise(fake_images) - fake_images).std(1)).mean() < 1e-4
assert torch.abs((inject_noise(fake_images) - fake_images).std(2)).mean() > 1e-4
assert torch.abs((inject_noise(fake_images) - fake_images).std(3)).mean() > 1e-4

# Check that the per-channel change is roughly normal
per_channel_change = (inject_noise(fake_images) - fake_images).mean(1).std()
print(per_channel_change)
assert per_channel_change > 0.9 and per_channel_change < 1.1
inject_noise.weight = torch.nn.Parameter(torch.zeros_like(inject_noise.weight))
assert torch.abs((inject_noise(fake_images) - fake_images)).mean() < 1e-4
assert len(inject_noise.weight.shape) == 4
print("Success!!!")


# UNQ_C4 (UNIQUE CELL IDENTIFIER,DO NOT EDIT)
# GRADED CELL:AdaIN

class AdaIN(torch.nn.Module):
    '''
    AdaIN Class
    VALUES:
        channels: the number of channels the image has ,a scalar
        w_dim:the dimension of the
    '''

    def __init__(self, channels, w_dim):
        super().__init__()
        # Normalize the input per-dimension
        self.instance_norm = torch.nn.InstanceNorm2d(channels)
        self.style_scale_transform = torch.nn.Linear(w_dim, channels)
        self.style_shift_transform = torch.nn.Linear(w_dim, channels)

    def forward(self, image, w):
        normalized_image = self.instance_norm(image)
        a = self.style_scale_transform(w)[:, :, None, None]
        style_scale = self.style_scale_transform(w)[:, :, None, None]
        style_shift = self.style_shift_transform(w)[:, :, None, None]
        # Calculate the transformed image
        transformed_image = style_scale * normalized_image + style_shift
        return transformed_image

    # UNIT TEST COMMENT:Required for grading
    def get_style_scale_transform(self):
        return self.style_scale_transform

    # UNIT TEST COMMENT: Required for grading
    def get_style_shift_transform(self):
        return self.style_shift_transform

    # UNIT TEST COMMENT:Required for grading
    def get_self(self):
        return self


w_channels = 50
image_channels = 20
image_size = 30
n_test = 10
adain = AdaIN(image_channels, w_channels)
test_w = torch.randn(n_test, w_channels)

a = adain.style_scale_transform(test_w)
b = adain.style_shift_transform(test_w)

assert adain.style_scale_transform(test_w).shape == adain.style_scale_transform(test_w).shape
assert adain.style_scale_transform(test_w).shape[-1] == image_channels
assert tuple(adain(torch.randn(n_test, image_channels, image_size, image_size), test_w).shape) == (
    n_test, image_channels, image_size, image_size)
print("Success3!!")

w_channels = 3
image_channels = 2
image_size = 3
n_test = 1
adain = AdaIN(image_channels, w_channels)
adain.style_scale_transform.weight.data = torch.ones_like(adain.style_scale_transform.weight.data) / 4
adain.style_scale_transform.bias.data = torch.zeros_like(adain.style_scale_transform.bias.data)

adain.style_shift_transform.weight.data = torch.ones_like(adain.style_shift_transform.weight.data) / 5
adain.style_shift_transform.bias.data = torch.zeros_like(adain.style_shift_transform.bias.data)

test_input = torch.ones(n_test, image_channels, image_size, image_size)

test_input[:, :, 0] = 0
test_w = torch.ones(n_test, w_channels)
test_output = adain(test_input, test_w)

assert torch.abs(test_output[0, 0, 0, 0] - 3 / 5 + torch.sqrt(torch.tensor(9 / 8))) < 1e-4
assert torch.abs(test_output[0, 0, 1, 0] - 3 / 5 - torch.sqrt(torch.tensor(9 / 32))) < 1e-4

print("Success !! 4")


# UNQ_C5 (UNIQUE CELL IDENTIFIER,DO NOT EDIT)
# GRADED CELL: MicroStyleGANGeneratorBlock

class MicroStyleGANGeneratorBlock(torch.nn.Module):
    def __init__(self, in_chan, out_chan, w_dim, kernel_size, starting_size, use_upsample=True):
        super().__init__()
        self.use_upsample = use_upsample
        # Replace the Nones in order to:
        # 1.Upsample to the starting_size,bilinear(https)
        # 2.Create a kernel_size convolution which takes in
        #    an image with in_chan and outputs one with out_chan
        # 3.Create an object to inject noise
        # 4.Create an AdaIN object
        # 5. create a LeakyRelu activation with slope 0.2
        ####START CODE HERE ####
        if self.use_upsample:
            self.upsample = torch.nn.Upsample((starting_size), mode="bilinear")
        self.conv = torch.nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=kernel_size, padding=1)
        self.inject_noise = InjectNoise(out_chan)
        self.adain = AdaIN(out_chan, w_dim)
        self.activation = torch.nn.LeakyReLU(0.2)

    def forward(self, x, w):
        if self.use_upsample:
            x = self.upsample(x)
        x = self.conv(x)
        x = self.inject_noise(x)
        x = self.activation(x)
        x = self.adain(x, w)
        return x

    def get_self(self):
        return self


test_stylegan_block = MicroStyleGANGeneratorBlock(in_chan=128, out_chan=64, w_dim=256, kernel_size=3, starting_size=8)
test_x = torch.ones(1, 128, 4, 4)
test_x[:, :, 1:3, 1:3] = 0
test_w = torch.ones(1, 256)
test_x = test_stylegan_block.upsample(test_x)
assert tuple(test_x.shape) == (1, 128, 8, 8)
assert torch.abs(test_x.mean() - 0.75) < 1e-4
test_x = test_stylegan_block.conv(test_x)
assert tuple(test_x.shape) == (1, 64, 8, 8)
test_x = test_stylegan_block.inject_noise(test_x)
test_x = test_stylegan_block.activation(test_x)
assert test_x.min() < 0
assert -test_x.min() / test_x.max() < 0.4
test_x = test_stylegan_block.adain(test_x, test_w)
foo = test_stylegan_block(torch.ones(10, 128, 4, 4), torch.ones(10, 256))

print("Success!")


class MicroStyleGANGenerator(torch.nn.Module):
    def __init__(self,
                 z_dim,
                 map_hidden_dim,
                 w_dim,
                 in_chan,
                 out_chan,
                 kernel_size,
                 hidden_chan):
        super().__init__()
        self.map = MappingLayers(z_dim, map_hidden_dim, w_dim)
        self.starting_constant = torch.nn.Parameter(torch.randn(1, in_chan, 4, 4))
        self.block0 = MicroStyleGANGeneratorBlock(in_chan, hidden_chan, w_dim, kernel_size, 4, use_upsample=False)
        self.block1 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 8)
        self.block2 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 16)

        self.block1_to_image = torch.nn.Conv2d(hidden_chan, out_chan, kernel_size=1)
        self.block2_to_image = torch.nn.Conv2d(hidden_chan, out_chan, kernel_size=1)
        self.alpha = 0.2

    def upsample_to_match_size(self, samller_image, bigger_image):
        return torch.nn.functional.interpolate(samller_image, size=bigger_image.shape[-2:], mode='bilinear')

    def forward(self, noise, return_intermediate=False):
        '''
                Function for completing a forward pass of MicroStyleGANGenerator: Given noise,
                computes a StyleGAN iteration.
                Parameters:
                    noise: a noise tensor with dimensions (n_samples, z_dim)
                    return_intermediate: a boolean, true to return the images as well (for testing) and false otherwise
                '''
        x = self.starting_constant
        w = self.map(noise)
        x = self.block0(x, w)
        x_small = self.block1(x, w)  # First generator run output
        x_small_image = self.block1_to_image(x_small)
        x_big = self.block2(x_small, w)  # Second generator run output
        x_big_image = self.block2_to_image(x_big)
        x_small_upsample = self.upsample_to_match_size(x_small_image,
                                                       x_big_image)  # Upsample first generator run output to be same size as second generator run output
        # Interpolate between the upsampled image and the image from the generator using alpha

        #### START CODE HERE ####
        interpolation = self.alpha * (x_big_image) + (1 - self.alpha) * (x_small_upsample)
        #### END CODE HERE ####

        if return_intermediate:
            return interpolation, x_small_upsample, x_big_image
        return interpolation

    # UNIT TEST COMMENT: Required for grading
    def get_self(self):
        return self


z_dim = 128
out_chan = 3
truncation = 0.7

mu_stylegan = MicroStyleGANGenerator(
    z_dim=z_dim,
    map_hidden_dim=1024,
    w_dim=496,
    in_chan=512,
    out_chan=out_chan,
    kernel_size=3,
    hidden_chan=256
)

test_samples = 10
# test_result = mu_stylegan(get_truncated_noise(test_samples, z_dim, truncation))
a = get_truncated_noise(test_samples, z_dim, truncation)
test_result = mu_stylegan(a)

# Check if the block works
assert tuple(test_result.shape) == (test_samples, out_chan, 16, 16)

# Check that the interpolation is correct
mu_stylegan.alpha = 1.
test_result, _, test_big = mu_stylegan(
    get_truncated_noise(test_samples, z_dim, truncation),
    return_intermediate=True)
assert torch.abs(test_result - test_big).mean() < 0.001
mu_stylegan.alpha = 0.
test_result, test_small, _ = mu_stylegan(
    get_truncated_noise(test_samples, z_dim, truncation),
    return_intermediate=True)
assert torch.abs(test_result - test_small).mean() < 0.001
print("Success!")
