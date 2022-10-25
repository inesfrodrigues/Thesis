import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.autograd import Variable

from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm


class CNN(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=2, padding=1): # kernel = 3, stride = 2, padding = 1
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
        self.bn1 = nn.GroupNorm(1, 16),
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = dim, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
        self.bn2 = nn.GroupNorm(1, dim),
        self.silu = nn.SiLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.silu(x)
        x = self.conv2(x)
        x = self.bn2(x) #ver
        x = self.silu(x) #ver
        
        return x
    

class Downsample(nn.Module): #page 4, paragraph 2
    def __init__(self, in_channels, out_channels): 
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, stride = 2, bias = False)
        self.bn = nn.GroupNorm(1, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        
        return x


class HMHSA(nn.Module):
    def __init__(self, dim, head, grid_size, ds_ratio, drop): # we have an embedding and we are going to split it in (heads) =! parts
        super().__init__()
        self.num_heads = dim // head
        self.grid_size = grid_size
        self.head = head
        self.dim = dim
        
        assert (self.num_heads * head == dim), "Dim needs to be divisible by Head."
        
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias = False) # q, k, v are all the same; * 3 bc 3 vectors qkv
        self.proj = nn.Conv2d(dim, dim, 1, bias = False) 
        self.norm = nn.GroupNorm(1, dim)
        self.drop = nn.Dropout2d(drop, inplace = True)

        if self.grid_size > 1:
            self.norm = nn.GroupNorm(1, dim)
            self.avg_pool = nn.AvgPool2d(ds_ratio, stride = ds_ratio)
            self.q = nn.Conv2d(dim, dim, 1, bias = False)
            self.kv = nn.Conv2d(dim, dim * 2, 1, bias = False) # * 2 bc kv
        
    def forward(self, x): # mask
        N, C, H, W = x.shape # N - nº samples, C, H, W - feature dimension, height, width of x (see paper)
        # qkv = self.qkv(x) # do "linear"
        qkv = self.qkv(self.norm(x))

        if self.grid_size > 1:

            # formula (6)
            grid_h, grid_w = H // self.grid_size, W // self.grid_size # grid_h - H/G_1; grid_w - W/G_1 -> paper(6)
            qkv = qkv.reshape(N, 3, self.num_heads, self.head, grid_h, self.grid_size, grid_w, self.grid_size) # 3 bc qkv; head=C; grid_h*grid_size=H... -> paper(6)
            qkv = qkv.permute(1, 0, 2, 4, 6, 5, 7, 3) # (3, N, num_heads, grid_h, grid_w, grid_size, grid_size, head) -> paper(6) 2nd eq.
            qkv = qkv.reshape(3, -1, self.grid_size * self.grid_size, self.head) # -1 -> single dim --- DUV WHY --- -> reshape to paper(6) 2nd eq.
            query, key, value = qkv[0], qkv[1], qkv[2]
        
            # eq. (2)
            attention = query @ key.transpose(-2, -1)

            #if mask is not None:
                #attention = attention.masked_fill(mask = 0, value = float("-1e20"))

            attention = torch.softmax(attention / (self.dim ** (1/2)), dim = -1)

            # formula (8)
            attention_x = (attention @ value).reshape(N, self.num_heads, grid_h, grid_w, self.grid_size, self.grid_size, self.head)
            attention_x = attention_x.permute(0, 1, 6, 2, 4, 3, 5).reshape(N, C, H, W) # (N, num_heads, head, grid_h, grid_size, grid_w, grid_size); reshape -> concatenate


            #formula (9)
            attention_x = self.norm(x + attention_x)

            # formula (10)
            #kv = self.kv(self.avg_pool(attention_x))
            kv = self.kv(self.norm(self.avg_pool(attention_x)))

            # formula (11)(12)
            query = self.q(attention_x).reshape(N, self.num_heads, self.head, -1)
            query = query.transpose(-2, -1) # (N, num_heads, -1, head) 
            kv = kv.reshape(N, 2, self.num_heads, self.head, -1)
            kv = kv.permute(1, 0, 2, 4, 3) # (2, N, num_heads, -1, head)
            key, value = kv[0], kv[1]

        else:
            qkv = qkv.reshape(N, 3, self.num_heads, self.head, -1)
            qkv = qkv.permute(1, 0, 2, 4, 3) # (2, N, num_heads, -1, head)
            query, key, value = qkv[0], qkv[1], qkv[2]  

        # eq. (2)
        attention = query @ key.transpose(-2, -1)

        #if mask is not None:
        #        attention = attention.masked_fill(mask = 0, value = float("-1e20"))
        # do masks

        attention = torch.softmax(attention / (self.dim ** (1/2)), dim = -1)

        # formula (13)
        global_attention_x = (attention @ value).transpose(-2, -1).reshape(N, C, H, W) # concatenate


        # formula (14)
        if self.grid_size > 1:
            global_attention_x = global_attention_x + attention_x

        x = self.drop(self.proj(global_attention_x)) # x + ...
        
        return x


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, kernel_size, drop = 0):
        expanded_channels = in_channels * expansion
        padding = (kernel_size - 1) // 2
        # use ResidualAdd if dims match, otherwise a normal Sequential
        super().__init__()
        # narrow -> wide
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, kernel_size = 1, padding = 0, bias = False),
            nn.GroupNorm(1, expanded_channels),
            nn.SiLU()
        )                
        # wide -> wide
        self.conv2 = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size = kernel_size, padding = padding, groups = expanded_channels, bias = False),
            nn.GroupNorm(1, expanded_channels),
            nn.SiLU()
        )
        # wide -> narrow
        self.conv3 = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size = 1, padding = 0, bias = False),
            nn.GroupNorm(1, out_channels)
        )

        self.drop = nn.Dropout2d(drop, inplace = True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.drop(x)

        return x
    
    
 class Block(nn.Module):
    def __init__(self, dim, head, kernel_size, expansion, grid_size, ds_ratio, drop, drop_path):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.HMSHA = HMHSA(dim = dim, head = head, grid_size = grid_size, ds_ratio = ds_ratio, drop = drop)
        self.MLP = MBConv(in_channels = dim, out_channels = dim, expansion = expansion, kernel_size = kernel_size)

    def forward(self, x):
        # x = x + self.HMSHA(x)
        # x = x + self.MLP(x)

        x = self.drop_path(self.HMSHA(x))
        x = self.drop_path(self.MLP(x))

        return x
    
    
    
  class HAT_Net(nn.Module):
    def __init__(self, dims, head, kernel_sizes, expansions, grid_sizes, ds_ratios, drop, depths, fc, drop_path_rate, act_layer = nn.SiLU):
        super().__init__()
        self.depths = depths

        # two sequential vanilla 3 × 3 convolutions - first downsample
        #self.CNN = CNN(dim = dims[0])
        self.CNN = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 2, padding = 1, bias = False),
            nn.GroupNorm(1, 16),
            act_layer(inplace = True),
            nn.Conv2d(in_channels = 16, out_channels = dims[0], kernel_size = 3, stride = 2, padding = 1, bias = False),
            nn.GroupNorm(1, dims[0]),
            act_layer(inplace = True),
        )

        # block - H-MSHA + MLP
        self.blocks = []
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for stage in range(len(dims)):
            self.blocks.append(nn.ModuleList([Block(
                dim = dims[stage], head = head, kernel_size = kernel_sizes[stage], expansion = expansions[stage],
                grid_size = grid_sizes[stage], ds_ratio = ds_ratios[stage], drop = drop, drop_path = dpr[sum(depths[:stage]) + i])
                for i in range(depths[stage])])) # will calculate each block depth times
        self.blocks = nn.ModuleList(self.blocks)

        # downsamples
        self.ds1 = Downsample(in_channels = dims[0], out_channels = dims[1])
        self.ds2 = Downsample(in_channels = dims[1], out_channels = dims[2])
        self.ds3 = Downsample(in_channels = dims[2], out_channels = dims[3])

        # fully connected layer -> 1000
        self.fullyconnected = nn.Sequential(
            nn.Dropout(0.2, inplace = True),
            nn.Linear(dims[3], fc),
        )
                
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std = .02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.CNN(x)
        for block in self.blocks[0]:
            x = block(x)
        x = self.ds1(x)
        for block in self.blocks[1]:
            x = block(x)
        x = self.ds2(x)
        for block in self.blocks[2]:
            x = block(x)
        x = self.ds3(x)
        for block in self.blocks[3]:
            x = block(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1) # F so we specify the input
        # In adaptive_avg_pool2d, we define the output size we require at the end of the pooling operation, and pytorch infers what pooling parameters to use to do that.
        x = self.fullyconnected(x)

        return x
    
    
# TRAINING
    
model = HAT_Net(dims = [64, 128, 320, 512], head = 64, kernel_sizes = [5, 3, 5, 3], expansions = [8, 8, 4, 4],
        grid_sizes = [8, 7, 7, 1], ds_ratios = [8, 4, 2, 1], drop = 0, depths = [3, 6, 18, 3], fc = 1000, drop_path_rate = 0)
    
    

# Loading and normalizing the data.
# Define transformations for the training and test sets
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),
])

# CIFAR10 dataset consists of 50K training images. We define the batch size of 10 to load 5,000 batches of images.
batch_size = 10
number_of_labels = 10 

# Create an instance for training. 
# When we run this code for the first time, the CIFAR10 train dataset will be downloaded locally. 
train_set =CIFAR10(root = "./data", train = True, transform = preprocess, download = True)

# Create a loader for the training set which will read the data within batch size and put into memory.
train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 0)
print("The number of images in a training set is: ", len(train_loader) * batch_size)

# Create an instance for testing, note that train is set to False.
# When we run this code for the first time, the CIFAR10 test dataset will be downloaded locally. 
test_set = CIFAR10(root = "./data", train = False, transform = preprocess, download = True)

# Create a loader for the test set which will read the data within batch size and put into memory. 
# Note that each shuffle is set to false for the test loader.
test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 0)
print("The number of images in a test set is: ", len(test_loader) * batch_size)

print("The number of batches per epoch is: ", len(train_loader))
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


 
# Define the loss function with Classification Cross-Entropy loss and an optimizer with AdamW optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr = 0.001, weight_decay = 0.0001)



# Function to save the model
def saveModel():
    path = "./myFirstModel.pth"
    torch.save(model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():
    
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)


# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):
    
    best_accuracy = 0.0

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):
            
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            #running_loss += loss.item()     # extract the loss value
            #if i % 10 == 9:    
                # print every 1000 (twice per epoch) 
            #    print('[%d, %5d] loss: %.3f' %
            #          (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
            #    running_loss = 0.0

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = testAccuracy()
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
        
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy
            
            

# Function to show the images
def imageshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Function to test the model with a batch of images and show the labels predictions
def testBatch():
    # get batch of images from the test DataLoader  
    images, labels = next(iter(test_loader))

    # show all images as one image grid
    imageshow(torchvision.utils.make_grid(images))
   
    # Show the real labels on the screen 
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]] 
                               for j in range(batch_size)))
  
    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)
    
    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)
    
    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] 
                              for j in range(batch_size)))
    
   

if __name__ == "__main__":
    
    # Let's build our model
    train(300)
    print('Finished Training')

    # Test which classes performed well
    testModelAccuracy()
    
    # Let's load the model we just created and test the accuracy per label
    model = Network()
    path = "myFirstModel.pth"
    model.load_state_dict(torch.load(path))

    # Test with batch of images
    testBatch()
