import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baremetal_nn import TracedModule

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# set the seed for reproducibility
torch.manual_seed(0)

torch.set_printoptions(precision=3)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1=nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2=nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(in_features=1024, out_features=10)

    def forward(self, input):
        #residual = input
        out = self.conv1(input)
        out = F.relu(out)
        out = self.conv2(out)  
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)
        residual = out
        out = self.conv3(out)
        out = F.relu(out, inplace=False)
        out = self.conv4(out)
        out = F.relu(out, inplace=False)
        out = out + residual
        out = self.conv5(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)
        out = self.conv6(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)
        residual = out
        out = self.conv7(out)
        out = F.relu(out, inplace=False)
        out = self.conv8(out)  
        out = F.relu(out, inplace=False)
        out = out + residual
        out = F.max_pool2d(out, 2, 2)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# prepare an example model
m = Net()

# set the model to evaluation mode
m.eval()

# wrap the model in our converter
m = TracedModule(m)

# we need to provide an example input to the model
# the value doesn't matter, as long as it has the right dimension and shape
test_input = torch.ones((1, 3, 32, 32))

'''data = unpickle('/vlsidl/projects1/re_am/python/networks/CIFAR10_ResNet18/data/cifar-10-batches-py/test_batch')
mean = np.array([0.0, 0.0, 0.0]).reshape(3, 1, 1)
std = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)

# Convert numpy array to torch tensor and reshape to [1, 3, 32, 32]
img = data[b'data'][0].reshape(3, 32, 32).astype(np.float32) / 255.0
img = (img - mean) / std
test_input = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # [1, 3, 32, 32]'''
#print(test_input.dtype)
#test_input = test_input.half()
#print(test_input.dtype)
print("Input:\n", test_input)
#print("first pixel in R:", data[b'data'][0][0] / (255.0*0.2023), "first pixel in G:", data[b'data'][0][1024] / (255.0*0.1994), "first pixel in B:", data[b'data'][0][2048] / (255.0*0.2010))

# run the model with the example input
with torch.no_grad():
    output = m.forward(test_input)
    print("output", output)
    print("output1", output[0,1016:1024])
    # tensor([[ 0.098,  0.041,  0.022, -0.045, -0.162, -0.034,  0.084,  0.035,  0.021,  0.102,  0.032,  0.047]])

conv_layer1 = m.model.conv1
fc_layer = m.model.fc

# Print weight and bias
print("Weight:\n", fc_layer.weight)
print("Weight:\n", fc_layer.weight[0,1021:1024])
#print("Bias:\n", conv_layer2.bias)
# now the converter knows the dimension and shape of each layer
# we can convert the model to our baremetal C runtime
m.convert()


test_input = test_input.permute(0, 2, 3, 1) # NCHW -> NHWC
zero_pad = np.zeros(test_input.shape[3])

arr = test_input.detach().cpu().numpy()
parts = []
for oh in range(test_input.shape[1]):
    for ow in range(test_input.shape[2]):
        for ih in [oh-1,oh,oh+1]:
            for iw in [ow-1,ow,ow+1]:
                if ih==-1 or iw==-1 or ih==(test_input.shape[1]) or iw==(test_input.shape[2]):
                    parts.append(zero_pad.ravel())
                else:
                    parts.append(arr[0,ih,iw,:].ravel())
        for z in range(37):
            parts.append(np.zeros(1))
flat = np.concatenate(parts)
    

img_content = b""
img_content += flat.astype(np.int8).tobytes()

with open('img.bin', "wb") as f:
    f.write(img_content)
# the converted model is saved in `model.h`
# and the weights are saved in `model.bin`
print("Done.")
