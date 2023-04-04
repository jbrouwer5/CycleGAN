



def imports():
    """
    Import PyTorch libraries.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    import matplotlib.pyplot as plt
    import numpy as np
    import random


def enable_cuda():
    """
    Enable CUDA if the GPU is available
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')



def dataloader():
    """
    Define the dataloader for the Fashion MNIST dataset.
    """
    train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=
                                                    transforms.Compose([transforms.ToTensor()]))
    test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=
                                                transforms.Compose([transforms.ToTensor()])) 
    train_loader = torch.utils.data.DataLoader(train_set, 
                                            batch_size=100)
    test_loader = torch.utils.data.DataLoader(test_set,
                                            batch_size=100)
    def output_label(label):
        output_mapping = {
                    0: "T-shirt/Top",
                    1: "Trouser",
                    2: "Pullover",
                    3: "Dress",
                    4: "Coat", 
                    5: "Sandal", 
                    6: "Shirt",
                    7: "Sneaker",
                    8: "Bag",
                    9: "Ankle Boot"
                    }
        input = (label.item() if type(label) == torch.Tensor else label)
        return output_mapping[input]
    image, label = next(iter(train_set))
    # print(image.shape[0], image.shape[1], image.shape[2])
    plt.imshow(image.squeeze(), cmap="gray")
    print(output_label(label))



"""
Define the generator
"""
class Generator(nn.Module):
    def __init__(self, gen_noise, image_channels, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x gen_noise x 1 x 1
            
            self._block(gen_noise, features_g * 16, 3, 1, 0), 
            self._block(features_g * 16, features_g * 8, 3, 2, 2), 
            self._block(features_g * 8, features_g * 4, 3, 2, 0), 
            self._block(features_g * 4, features_g * 2, 3, 2, 0),  
            nn.ConvTranspose2d(
                features_g * 2, image_channels, kernel_size=2, stride=2, padding=1
            ),
            # Output: N x image_channels x 28 * 28

            # Uses a tanh 
            nn.Tanh(),
        )

    # Predefined block that includes a convolution layer, activation function, 
    # and batch normalization
    def _block(self, input_channels, output_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                input_channels,
                output_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),

            nn.BatchNorm2d(output_channels),

            nn.ReLU() # Gen uses ReLU whereas disc uses a leaky ReLU
        )

    def forward(self, x):
        return self.net(x)

"""
Define the discriminator 
"""
class Discriminator(nn.Module):
  def __init__(self, color_channels, out_features):
    super(Discriminator, self).__init__()
    self.disc = nn.Sequential(
            # My input is channels(1) * 28 * 28 
            # First layer doesn't use batch
            nn.Conv2d(color_channels, out_features, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            self.disc_block(out_features, out_features * 2, 3, 2, 1),
            self.disc_block(out_features * 2, out_features * 4, 3, 2, 1),
            self.disc_block(out_features * 4, out_features * 8, 3, 2, 1),

            
            nn.Conv2d(out_features * 8, 1, kernel_size=2, stride=2, padding=0),
            nn.Sigmoid(),
        )
    
  def disc_block(self, input_channels, output_channels, kernel_size, stride, padding):
      return nn.Sequential(
          nn.Conv2d(
              input_channels,
              output_channels,
              kernel_size,
              stride,
              padding,
              bias=False,
          ),
          nn.BatchNorm2d(output_channels),
          nn.LeakyReLU(0.2),
      )

  def forward(self, x):
      return self.disc(x)



"""
Initialize the network and the Adam optimizer
"""
def initialize_weights(model):
  # Initializes weights according to the DCGAN paper
  for m in model.modules():
      if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
          nn.init.normal_(m.weight.data, 0.0, 0.02)


def initialize():
    # Hyperparameters
    LEARNING_RATE = 2e-4 
    BATCH_SIZE = 128
    IMAGE_SIZE = 28
    IMG_CHANNELS = 1
    NOISE_DIM = 100
    NUM_EPOCHS = 500
    FEATURES_DISC = 28
    FEATURES_GEN = 28

    # transforms = transforms.Compose(
    #     [
    #         transforms.Resize(IMAGE_SIZE),
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             [0.5 for _ in range(IMG_CHANNELS)], [0.5 for _ in range(IMG_CHANNELS)]
    #         ),
    #     ]
    # )

    gen = Generator(NOISE_DIM, IMG_CHANNELS, FEATURES_GEN).to(device)
    disc = Discriminator(IMG_CHANNELS, FEATURES_DISC).to(device)
    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.BCELoss().to(device)


# TEST the initialization of the gen and disc
def test():
    N, in_channels, H, W = 8, 1, 28, 28
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W)).to("cuda")
    # x = .Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

    # print(x.shape)
    disc = Discriminator(in_channels, 4).to("cuda")
    # print(disc)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    # print(gen)
    z = torch.randn((N, noise_dim, 1, 1))
    # print(z.shape)
    # print(gen(z).shape)
    # print((N, in_channels, H, W))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("Success, tests passed!")






def train():
  # Target labels not needed!  unsupervised
  for batch_idx, (real, _) in enumerate(train_loader):
    real = real.to(device)
    noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
    fake = gen(noise)

    ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
    disc_real = disc(real).reshape(-1)
    loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
    disc_fake = disc(fake.detach()).reshape(-1)
    loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
    loss_disc = (loss_disc_real + loss_disc_fake) / 2
    disc.zero_grad()
    loss_disc.backward()
    opt_disc.step()

    ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
    output = disc(fake).reshape(-1)
    loss_gen = criterion(output, torch.ones_like(output))
    gen.zero_grad()
    loss_gen.backward()
    opt_gen.step()

  print(epoch)
  print(f"Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}")
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots(5,5)
  ax = ax.reshape(-1)
  
  for i in range(25):
    ax[i].imshow(fake[i,0].cpu().data.numpy(), cmap = 'gray')
  plt.show()



if __name__ == "__main__":
    imports() 
    enable_cuda() 
    dataloader()
    initialize()
    for epoch in range(NUM_EPOCHS):
       train() 