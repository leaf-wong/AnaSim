import numpy as np
import torch
import torch.nn as nn

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
        self.actor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, input):
        output = self.actor.forward(input)
        return output


# prepare an example model
m = Net()

# set the model to evaluation mode
m.eval()

# wrap the model in our converter
m = TracedModule(m)

# we need to provide an example input to the model
# the value doesn't matter, as long as it has the right dimension and shape
#test_input = torch.ones((1024, )).unsqueeze(0)
data = unpickle('/vlsidl/projects1/re_am/python/networks/CIFAR10_ResNet18/data/cifar-10-batches-py/test_batch')
mean = np.array([0.0, 0.0, 0.0]).reshape(3, 1, 1)
std = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)

# Convert numpy array to torch tensor and reshape to [1, 3, 32, 32]
img = data[b'data'][0].reshape(3, 32, 32).astype(np.float32) / 255.0
img = (img - mean) / std
test_input = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # [1, 3, 32, 32]
#print(test_input.dtype)
#test_input = test_input.half()
#print(test_input.dtype)
print("Input:\n", test_input)
print("first pixel in R:", data[b'data'][0][0] / (255.0*0.2023), "first pixel in G:", data[b'data'][0][1024] / (255.0*0.1994), "first pixel in B:", data[b'data'][0][2048] / (255.0*0.2010))

# run the model with the example input
with torch.no_grad():
    output = m.forward(test_input)
    print("output", output)
    # tensor([[ 0.098,  0.041,  0.022, -0.045, -0.162, -0.034,  0.084,  0.035,  0.021,  0.102,  0.032,  0.047]])

conv_layer = m.model.actor[0]

# Print weight and bias
print("Weight:\n", conv_layer.weight)
print("Bias:\n", conv_layer.bias)
# now the converter knows the dimension and shape of each layer
# we can convert the model to our baremetal C runtime
m.convert()

# the converted model is saved in `model.h`
# and the weights are saved in `model.bin`
print("Done.")
