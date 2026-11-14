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


class ReorgLayer(nn.Module):
    """Move local spatial blocks into the channel dimension (YOLOv2 reorg)."""

    def __init__(self, stride=2):
        super().__init__()
        if stride < 1:
            raise ValueError("stride must be positive")
        self.stride = stride
        self._baremetal_leaf = True


    def forward(self, x):
        batch, channels, height, width = x.shape
        #if height % self.stride != 0 or width % self.stride != 0:
        #    raise ValueError("height and width must be divisible by stride")
        out_height = height // self.stride
        out_width = width // self.stride
        x = x.reshape(batch, channels, out_height, self.stride,
                      out_width, self.stride)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.reshape(batch, channels * self.stride * self.stride,
                         out_height, out_width) # 2nd demension order in c,h,w need change to h,w,c for hardware


def conv_bn_leaky(in_channels, out_channels, kernel_size, stride=1,
                  same_padding=False, relu=True):
    padding = (kernel_size - 1) // 2 if same_padding else 0
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                  padding=padding, bias=True),
        #nn.BatchNorm2d(out_channels, momentum=0.01),
    ]
    if relu:
        #layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def _make_layers(in_channels, net_cfg):
    layers = []

    if len(net_cfg) > 0 and isinstance(net_cfg[0], list):
        for sub_cfg in net_cfg:
            layer, in_channels = _make_layers(in_channels, sub_cfg)
            layers.append(layer)
    else:
        for item in net_cfg:
            if item == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels, ksize = item
                layers.append(conv_bn_leaky(in_channels, out_channels, ksize,
                                            same_padding=True))
                in_channels = out_channels

    return nn.Sequential(*layers), in_channels


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        net_cfgs = [
            # conv1s
            #[(32, 3)],
            [(32, 3)],
            ['M', (64, 3)],
            ['M', (128, 3), (64, 1), (128, 3)],
            ['M', (256, 3), (128, 1), (256, 3)],
            ['M', (512, 3), (256, 1), (512, 3), (256, 1), (512, 3)],
            # conv2
            ['M', (1024, 3), (512, 1), (1024, 3), (512, 1), (1024, 3)],
            # ------------
            # conv3
            [(1024, 3), (1024, 3)],
            # conv4
            [(1024, 3)]
        ]

        # darknet
        self.conv1s, c1 = _make_layers(3, net_cfgs[0:5])
        self.conv2, c2 = _make_layers(c1, net_cfgs[5])
        # ---
        self.conv3, c3 = _make_layers(c2, net_cfgs[6])

        stride = 2
        # stride*stride times the channels of conv1s
        self.reorg = ReorgLayer(stride=2)
        # cat [conv1s, conv3]
        self.conv4, c4 = _make_layers((c1*(stride*stride) + c3), net_cfgs[7])

        self.num_anchors = 5
        self.num_classes = 20
        out_channels = self.num_anchors * (self.num_classes + 5)
        self.conv5 = nn.Conv2d(c4, out_channels, 1, 1, bias=True)
        #self.global_average_pool = nn.AvgPool2d((1, 1)) #only for classfication to downsample. 
        

    def forward(self, input):
        #residual = input

        conv1s = self.conv1s(input)
        
        conv2 = self.conv2(conv1s)
        conv3 = self.conv3(conv2)
        #return conv3
        conv1s_reorg = self.reorg(conv1s)
        cat_1_3 = torch.cat([conv1s_reorg, conv3], 1)
        conv4 = self.conv4(cat_1_3)
        conv5 = self.conv5(conv4)   # batch_size, out_channels, h, w
        #print("conv5 shapes", conv5.shape)
        #return conv5
        #trace model stop here, decoding done on the host side
        

        #global_average_pool = self.global_average_pool(conv5)
        #print("global_average_pool shapes", global_average_pool.shape)
        # for detection
        # bsize, c, h, w -> bsize, h, w, c ->
        #                   bsize, h x w, num_anchors, 5+num_classes
        #bsize, _, h, w = global_average_pool.size()
        bsize, _, h, w = conv5.size()
        # assert bsize == 1, 'detection only support one image per batch'
        #global_average_pool_reshaped = global_average_pool.permute(
        global_average_pool_reshaped = conv5.permute(
            0, 2, 3, 1).contiguous().view(
                bsize, -1, self.num_anchors, self.num_classes + 5)

        # tx, ty, tw, th, to -> sig(tx), sig(ty), exp(tw), exp(th), sig(to)
        xy_pred = torch.sigmoid(global_average_pool_reshaped[:, :, :, 0:2])
        wh_pred = torch.exp(global_average_pool_reshaped[:, :, :, 2:4])
        bbox_pred = torch.cat([xy_pred, wh_pred], 3)
        iou_pred = torch.sigmoid(global_average_pool_reshaped[:, :, :, 4:5])

        score_pred = global_average_pool_reshaped[:, :, :, 5:].contiguous()
        prob_pred = F.softmax(score_pred.view(-1, score_pred.size()[-1]), dim=-1).view_as(score_pred)  # noqa

        return bbox_pred, iou_pred, prob_pred


# prepare an example model
m = Net()

# set the model to evaluation mode
m.eval()

# wrap the model in our converter
m = TracedModule(m)

'''image_w = 448
image_hardware_w = image_w // 4
imc_dim = 256
channels_per_lane = imc_dim // 4

input_array = np.zeros(
    (image_w, image_hardware_w, 4, channels_per_lane),
    dtype=np.int8,
)

for i in range(image_w):
    if i == 0:
        row_start, row_end = 1, 3
    elif i == image_w - 1:
        row_start, row_end = 0, 2
    else:
        row_start, row_end = 0, 3

    for j in range(image_hardware_w):
        for l in range(4):
            pixel_x = j * 4 + l

            if j == 0 and l == 0:
                col_start, col_end = 1, 3
            elif pixel_x == image_w - 1:
                col_start, col_end = 0, 2
            else:
                col_start, col_end = 0, 3

            for k in range(3):
                for m in range(3):
                    for n in range(3):
                        offset = k * 3 * 3 + m * 3 + n

                        if (
                            row_start <= k < row_end
                            and col_start <= m < col_end
                        ):
                            input_array[i, j, l, offset] = 1

input_array.tofile("input.bin")'''


# we need to provide an example input to the model
# the value doesn't matter, as long as it has the right dimension and shape
test_input = torch.ones((1, 3, 416, 416))

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
    #print("output", output)
    print("output shapes", [tensor.shape for tensor in output])

# Print weight and bias
#print("Weight:\n", fc_layer.weight)
#print("Weight:\n", fc_layer.weight[0,1021:1024])
#print("Bias:\n", conv_layer2.bias)
# now the converter knows the dimension and shape of each layer
# we can convert the model to our baremetal C runtime
m.convert()

psum = torch.zeros((1, 32, 416, 416), dtype=torch.float32)
conv_layer = m.model.conv1s[0][0][0]
#conv_layer = m.model.conv1s[1][1][0]
for i in range(3):
    for j in range(3):
        w_mask = torch.zeros_like(conv_layer.weight)
        #w_mask[:, :, i, j] = (conv_layer.weight[:, :, i, j]*630).round().int()  
        w_mask[:, :, i, j] = (conv_layer.weight[:, :, i, j]*157.5).round().int()  
        #print("weight[{}][{}]:\n".format(i, j), w_mask[:, :, i, j])
        partial_half = F.conv2d(test_input, w_mask, None, 1,'same')  
        psum = psum + partial_half
        #print("psum after conv with weight[{}][{}]:\n".format(i, j), psum)

#print(" bias:\n", (conv_layer.bias*630).round().int())
#print(" bias:\n", (conv_layer.bias*157.5).round().int())
psum = ((psum + (conv_layer.bias*157.5).round().int().view(1, 32, 1, 1)*1)/3).round().int()
#print("psum00 after bias:\n", psum[:, :, 0, 0])
#print("psum99 after bias:\n", psum[:, :, 9, 9])

psum = F.relu(psum, inplace=False) 
conv1s_1 = F.max_pool2d(psum, kernel_size=2, stride=2, padding=0, ceil_mode=False)

conv_layer = m.model.conv1s[1][1][0]
psum_conv1s_1_1 = F.conv2d(conv1s_1, (conv_layer.weight[:, :, :, :]*630).round().int(), (conv_layer.bias*630).round().int()*55, 1,'same')
psum_conv1s_1_1 = (psum_conv1s_1_1/630).round().int()
#print("psum00 after bias:\n", psum_conv1s_1_1[:, :, 0, 0])
#print("psum99 after bias:\n", psum_conv1s_1_1[:, :, 8, 180])

psum_conv1s_1_1 = F.relu(psum_conv1s_1_1, inplace=True) 
conv1s_2 = F.max_pool2d(psum_conv1s_1_1, kernel_size=2, stride=2, padding=0, ceil_mode=False)

conv_layer = m.model.conv1s[2][1][0]
psum_conv1s_2_1 = F.conv2d(conv1s_2, (conv_layer.weight[:, :, :, :]*630).round().int(), (conv_layer.bias*630).round().int()*55, 1,'same')
psum_conv1s_2_1 = (psum_conv1s_2_1/630).round().int()
#print("psum00 after bias:\n", psum_conv1s_2_1[:, :, 0, 0])
psum_conv1s_2_1 = F.relu(psum_conv1s_2_1, inplace=True) 

conv_layer = m.model.conv1s[2][2][0]
#print("weight:\n", (conv_layer.weight[:, :, :, :]*630).round().int())
#print(" bias:\n", (conv_layer.bias*630).round().int())
psum_conv1s_2_1 = F.conv2d(psum_conv1s_2_1, (conv_layer.weight[:, :, :, :]*630).round().int(), (conv_layer.bias*630).round().int()*55, 1,'same')
psum_conv1s_2_1 = (psum_conv1s_2_1/630).round().int()
psum_conv1s_2_1 = F.relu(psum_conv1s_2_1, inplace=True) 

conv_layer = m.model.conv1s[2][3][0]
psum_conv1s_2_1 = F.conv2d(psum_conv1s_2_1, (conv_layer.weight[:, :, :, :]*630).round().int(), (conv_layer.bias*630).round().int()*55, 1,'same')
psum_conv1s_2_1 = (psum_conv1s_2_1/630).round().int()
#print("psumll after bias:\n", psum_conv1s_2_1[:, :, 111, 111])
#print("psum00 after bias:\n", psum_conv1s_2_1[:, :, 0, 0])
psum_conv1s_2_1 = F.relu(psum_conv1s_2_1, inplace=True) 

psum_conv1s_3 = F.max_pool2d(psum_conv1s_2_1, kernel_size=2, stride=2, padding=0, ceil_mode=False)

conv_layer = m.model.conv1s[3][1][0]
psum_conv1s_3 = F.conv2d(psum_conv1s_3, (conv_layer.weight[:, :, :, :]*630).round().int(), (conv_layer.bias*630).round().int()*55, 1,'same')
psum_conv1s_3 = (psum_conv1s_3/630).round().int()
psum_conv1s_3 = F.relu(psum_conv1s_3, inplace=True) 

conv_layer = m.model.conv1s[3][2][0]
psum_conv1s_3 = F.conv2d(psum_conv1s_3, (conv_layer.weight[:, :, :, :]*630).round().int(), (conv_layer.bias*630).round().int()*55, 1,'same')
psum_conv1s_3 = (psum_conv1s_3/630).round().int()
psum_conv1s_3 = F.relu(psum_conv1s_3, inplace=True) 

conv_layer = m.model.conv1s[3][3][0]
psum_conv1s_3 = F.conv2d(psum_conv1s_3, (conv_layer.weight[:, :, :, :]*630).round().int(), (conv_layer.bias*630).round().int()*55, 1,'same')
psum_conv1s_3 = (psum_conv1s_3/630).round().int()
#print("psum00 after bias:\n", psum_conv1s_3[:, :, 0, 0])
psum_conv1s_3 = F.relu(psum_conv1s_3, inplace=True) 

psum_conv1s_4 = F.max_pool2d(psum_conv1s_3, kernel_size=2, stride=2, padding=0, ceil_mode=False)

conv_layer = m.model.conv1s[4][1][0]
psum_conv1s_4 = F.conv2d(psum_conv1s_4, (conv_layer.weight[:, :, :, :]*630).round().int(), (conv_layer.bias*630).round().int()*55, 1,'same')
psum_conv1s_4 = (psum_conv1s_4/630).round().int()
#print("psum00 after bias:\n", psum_conv1s_4[:, :, 0, 0])
psum_conv1s_4 = F.relu(psum_conv1s_4, inplace=True) 

conv_layer = m.model.conv1s[4][2][0]
psum_conv1s_4 = F.conv2d(psum_conv1s_4, (conv_layer.weight[:, :, :, :]*630).round().int(), (conv_layer.bias*630).round().int()*55, 1,'same')
psum_conv1s_4 = (psum_conv1s_4/630).round().int()
#print("psum00 after bias:\n", psum_conv1s_4[:, :, 0, 0])
psum_conv1s_4 = F.relu(psum_conv1s_4, inplace=True) 

conv_layer = m.model.conv1s[4][3][0]
psum_conv1s_4 = F.conv2d(psum_conv1s_4, (conv_layer.weight[:, :, :, :]*630).round().int(), (conv_layer.bias*630).round().int()*55, 1,'same')
psum_conv1s_4 = (psum_conv1s_4/630).round().int()
psum_conv1s_4 = F.relu(psum_conv1s_4, inplace=True) 

conv_layer = m.model.conv1s[4][4][0]
psum_conv1s_4 = F.conv2d(psum_conv1s_4, (conv_layer.weight[:, :, :, :]*630).round().int(), (conv_layer.bias*630).round().int()*55, 1,'same')
psum_conv1s_4 = (psum_conv1s_4/630).round().int()
psum_conv1s_4 = F.relu(psum_conv1s_4, inplace=True) 

conv_layer = m.model.conv1s[4][5][0]
psum_conv1s_4 = F.conv2d(psum_conv1s_4, (conv_layer.weight[:, :, :, :]*630).round().int(), (conv_layer.bias*630).round().int()*55, 1,'same')
psum_conv1s_4 = (psum_conv1s_4/630).round().int()
#print("psum00 after bias:\n", psum_conv1s_4[:, :, 0, 0])
psum_conv1s_4 = F.relu(psum_conv1s_4, inplace=True) 

psum_conv2 = F.max_pool2d(psum_conv1s_4, kernel_size=2, stride=2, padding=0, ceil_mode=False)

conv_layer = m.model.conv2[1][0]
#print("weight:\n", (conv_layer.weight[:, :, :, :]*630).round().int())
#print(" bias:\n", (conv_layer.bias[256:512]*630).round().int())
psum_conv2 = F.conv2d(psum_conv2, (conv_layer.weight[:, :, :, :]*630).round().int(), (conv_layer.bias*630).round().int()*55, 1,'same')
psum_conv2 = (psum_conv2/200).round().int()
#print("psum00 after bias:\n", psum_conv2[:, 0:128, 0, 0])
psum_conv2 = F.relu(psum_conv2, inplace=True) 

conv_layer = m.model.conv2[2][0]
psum_conv2 = F.conv2d(psum_conv2, (conv_layer.weight[:, :, :, :]*630).round().int(), (conv_layer.bias*630).round().int()*55, 1,'same')
psum_conv2 = (psum_conv2/200).round().int()
psum_conv2 = F.relu(psum_conv2, inplace=True) 

conv_layer = m.model.conv2[3][0]
psum_conv2 = F.conv2d(psum_conv2, (conv_layer.weight[:, :, :, :]*630).round().int(), (conv_layer.bias*630).round().int()*55, 1,'same')
psum_conv2 = (psum_conv2/200).round().int()
psum_conv2 = F.relu(psum_conv2, inplace=True) 

conv_layer = m.model.conv2[4][0]
psum_conv2 = F.conv2d(psum_conv2, (conv_layer.weight[:, :, :, :]*630).round().int(), (conv_layer.bias*630).round().int()*55, 1,'same')
psum_conv2 = (psum_conv2/200).round().int()
psum_conv2 = F.relu(psum_conv2, inplace=True) 

conv_layer = m.model.conv2[5][0]
psum_conv2 = F.conv2d(psum_conv2, (conv_layer.weight[:, :, :, :]*630).round().int(), (conv_layer.bias*630).round().int()*55, 1,'same')
psum_conv2 = (psum_conv2/200).round().int()
#print("psum00 after bias:\n", psum_conv2[:, 0:128, 0, 0])
psum_conv2 = F.relu(psum_conv2, inplace=True) 

conv_layer = m.model.conv3[0][0]
psum_conv3 = F.conv2d(psum_conv2, (conv_layer.weight[:, :, :, :]*630).round().int(), (conv_layer.bias*630).round().int()*55, 1,'same')
psum_conv3 = (psum_conv3/200).round().int()
#print("psum00 after bias:\n", psum_conv3[:, 0:128, 0, 0])
psum_conv3 = F.relu(psum_conv3, inplace=True) 

conv_layer = m.model.conv3[1][0]
#print("weight:\n", (conv_layer.weight[256:512, :, :, :]*630).round().int())
#print(" bias:\n", (conv_layer.bias[256:512]*630).round().int())
psum_conv3 = F.conv2d(psum_conv2, (conv_layer.weight[:, :, :, :]*630).round().int(), (conv_layer.bias*630).round().int()*55, 1,'same')
psum_conv3 = (psum_conv3/200).round().int()
print("psum00 after bias:\n", psum_conv3[:, 0:128, 0, 0])
psum_conv3 = F.relu(psum_conv3, inplace=True) 

#print("psum00 before org:\n", psum_conv1s_4[:, 0:128, 0:2, 0:2])
psum_conv1s_4 = m.model.reorg(psum_conv1s_4)
#print("psum00 after org:\n", psum_conv1s_4[:, 0:128, 0, 0])
cat_1_3_test = torch.cat([psum_conv1s_4, psum_conv3], 1)

conv_layer = m.model.conv4[0][0]
print("weight:\n", (conv_layer.weight[0, 0:128, :, :]*630).round().int())
psum_conv4 = F.conv2d(cat_1_3_test, (conv_layer.weight[:, :, :, :]*630).round().int(), (conv_layer.bias*630).round().int()*55, 1,'same')
psum_conv4 = (psum_conv4/200).round().int()
print("psum22 after bias:\n", psum_conv4[:, 0:128, 2, 2])
psum_conv4 = F.relu(psum_conv4, inplace=True) 

conv_layer = m.model.conv5
print("weight:\n", (conv_layer.weight[0, 0:128, :, :]*630).round().int())
psum_conv5 = F.conv2d(psum_conv4, (conv_layer.weight[:, :, :, :]*630).round().int(), (conv_layer.bias*630).round().int()*55, 1,'same')
psum_conv5 = (psum_conv5/200).round().int()
print("psum00 after bias:\n", psum_conv5[:, 0:128, 0, 0])

# the converted model is saved in `model.h`
# and the weights are saved in `model.bin`
print("Done.")
