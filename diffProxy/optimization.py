import os
import imageio

import numpy as np
from numpy import random
import torch.nn as nn
import torch.nn.functional as F

import torch
from sbs.differentiable_generator import StyleGANCond
from utils import read_image

os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


################## ARGUMENTS ##############################
import argparse
parser = argparse.ArgumentParser(description="arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input", required=True, help="input image")
parser.add_argument("--params", required=True, help="txt file with the params")

args = vars(parser.parse_args())

input = args["input"]
params = args["params"]

if not os.path.exists('results/'):
    	os.makedirs('results/')

if '/' in input:
	name = input[input.rindex('/')+1:input.rindex('.')]
else:
	name = input[:input.rindex('.')]

if not os.path.exists('results/' + name):
    	os.makedirs('results/' + name)

with open(params) as f:
	paramsValues = f.readlines()[0]
	paramsValues = paramsValues.split()


generator_name = 'pptbf'
model_path = 'diffProxy.pkl'


init = {'method': 'avg'}
G = StyleGANCond(generator_name, model_path, init, model_type='norm')

real_np = read_image(input)

tiling = int(paramsValues[0])
zoom = float(paramsValues[2])
alpha = float(paramsValues[3])
jitter = float(paramsValues[1])
normblend = float(paramsValues[4])
wsmooth = float(paramsValues[5])
winfeatcorrel = float(paramsValues[6])
larp = float(paramsValues[10])
feataniso = float(paramsValues[7])
sigcos = float(paramsValues[8])
deltaorient = float(paramsValues[9])
amp = random.uniform(0.0, 0.15)
rx = float(random.randint(0, 10))
ry = float(random.randint(0, 10))
tx = random.uniform(-zoom, zoom)
ty = random.uniform(-zoom, zoom)

parameters = np.concatenate((tiling, zoom, alpha, jitter, normblend, wsmooth, winfeatcorrel, larp, feataniso, sigcos, deltaorient, amp, rx, ry, tx, ty), axis=None)
parameters = torch.as_tensor(parameters, dtype=torch.float64, device=G.device).unsqueeze(0)
parameters = torch.nn.parameter.Parameter(parameters)

G.set_params(parameters)
z = np.zeros((1, 512))
z = torch.from_numpy(z).float().to('cuda:0')
fake = G(z).detach().squeeze().cpu().numpy()
fake = fake[np.newaxis, ...]


def saveImage(filename, image):
    imageTMP = np.clip(image * 255.0, 0, 255).astype('uint8')
    imageio.imwrite(filename, imageTMP)

SCALING_FACTOR = 1

class VGG19(torch.nn.Module):

    def __init__(self):
        super(VGG19, self).__init__()

        self.block1_conv1 = torch.nn.Conv2d(3, 64, (3,3), padding=(1,1), padding_mode='reflect')
        self.block1_conv2 = torch.nn.Conv2d(64, 64, (3,3), padding=(1,1), padding_mode='reflect')

        self.block2_conv1 = torch.nn.Conv2d(64, 128, (3,3), padding=(1,1), padding_mode='reflect')
        self.block2_conv2 = torch.nn.Conv2d(128, 128, (3,3), padding=(1,1), padding_mode='reflect')

        self.block3_conv1 = torch.nn.Conv2d(128, 256, (3,3), padding=(1,1), padding_mode='reflect')
        self.block3_conv2 = torch.nn.Conv2d(256, 256, (3,3), padding=(1,1), padding_mode='reflect')
        self.block3_conv3 = torch.nn.Conv2d(256, 256, (3,3), padding=(1,1), padding_mode='reflect')
        self.block3_conv4 = torch.nn.Conv2d(256, 256, (3,3), padding=(1,1), padding_mode='reflect')

        self.block4_conv1 = torch.nn.Conv2d(256, 512, (3,3), padding=(1,1), padding_mode='reflect')
        self.block4_conv2 = torch.nn.Conv2d(512, 512, (3,3), padding=(1,1), padding_mode='reflect')
        self.block4_conv3 = torch.nn.Conv2d(512, 512, (3,3), padding=(1,1), padding_mode='reflect')
        self.block4_conv4 = torch.nn.Conv2d(512, 512, (3,3), padding=(1,1), padding_mode='reflect')

        self.relu = torch.nn.ReLU(inplace=True)
        self.downsampling = torch.nn.AvgPool2d((2,2))

    def forward(self, image):
        
        # RGB to BGR
        image = image[:, [2,1,0], :, :]

        # [0, 1] --> [0, 255]
        image = 255 * image

        # remove average color
        image[:,0,:,:] -= 103.939
        image[:,1,:,:] -= 116.779
        image[:,2,:,:] -= 123.68

        # block1
        block1_conv1 = self.relu(self.block1_conv1(image))
        block1_conv2 = self.relu(self.block1_conv2(block1_conv1))
        block1_pool = self.downsampling(block1_conv2)

        # block2
        block2_conv1 = self.relu(self.block2_conv1(block1_pool))
        block2_conv2 = self.relu(self.block2_conv2(block2_conv1))
        block2_pool = self.downsampling(block2_conv2)

        # block3
        block3_conv1 = self.relu(self.block3_conv1(block2_pool))
        block3_conv2 = self.relu(self.block3_conv2(block3_conv1))
        block3_conv3 = self.relu(self.block3_conv3(block3_conv2))
        block3_conv4 = self.relu(self.block3_conv4(block3_conv3))
        block3_pool = self.downsampling(block3_conv4)

        # block4
        block4_conv1 = self.relu(self.block4_conv1(block3_pool))
        block4_conv2 = self.relu(self.block4_conv2(block4_conv1))
        block4_conv3 = self.relu(self.block4_conv3(block4_conv2))
        block4_conv4 = self.relu(self.block4_conv4(block4_conv3))

        return[block1_conv1, block1_conv2, block2_conv1, block2_conv2, block3_conv1, block3_conv2, block3_conv3, block3_conv4, block4_conv1, block4_conv2, block4_conv3, block4_conv4]
        #return[block1_conv2, block2_conv2, block3_conv4]


vgg = VGG19().to(torch.device("cuda:0"))
vgg.load_state_dict(torch.load("vgg19.pth"))

#######################################################################
# Initialize optimized texture
#######################################################################

image_example = real_np[np.newaxis, ...]
image_example = np.swapaxes(image_example, 1, 3)
image_example = torch.from_numpy(image_example)
image_example = image_example.to(torch.device("cuda:0"))


#######################################################################
# LBFGS optimization with the slicing loss
#######################################################################

optimizer = torch.optim.Adam([parameters], lr=0.5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

def slicing_loss(params, image_example):

    with torch.no_grad():
        params[0][0] = int(paramsValues[0])
        params[0][1] = params[0][1].clamp(1.0, 20.0)
        params[0][2] = params[0][2].clamp(0.0, 2.0)
        params[0][3] = params[0][3].clamp(0.01, 0.5)
        params[0][4] = params[0][4].clamp(0.0, 1.0)
        params[0][5] = params[0][5].clamp(0.0, 0.99)
        params[0][6] = params[0][6].clamp(0.0, 1.0)
        params[0][7] = params[0][7].clamp(0.0, 1.0)
        params[0][8] = params[0][8].clamp(0.0, 5.0)
        params[0][9] = params[0][9].clamp(0.0, 10.0)
        params[0][10] = params[0][10].clamp(-0.1*np.pi, 0.1*np.pi)
        params[0][11] = params[0][11].clamp(0.0, 0.15)
        params[0][12] = params[0][12].clamp(0.0, 10)
        params[0][13] = params[0][13].clamp(0.0, 10)
        params[0][14] = params[0][14].clamp(-params[0][1], params[0][1])
        params[0][15] = params[0][15].clamp(-params[0][1], params[0][1])

    #print(params)

    G.set_params(params)
    image_optimized = G(z)

    # generate VGG19 activations
    list_activations_generated = vgg(image_optimized)
    list_activations_example   = vgg(image_example)
    
    # iterate over layers
    loss = 0
    for l in range(len(list_activations_example)):
        # get dimensions
        b = list_activations_example[l].shape[0]
        dim = list_activations_example[l].shape[1]
        n = list_activations_example[l].shape[2]*list_activations_example[l].shape[3]
        # linearize layer activations and duplicate example activations according to scaling factor
        activations_example = list_activations_example[l].view(b, dim, n).repeat(1, 1, SCALING_FACTOR*SCALING_FACTOR)
        activations_generated = list_activations_generated[l].view(b, dim, n*SCALING_FACTOR*SCALING_FACTOR)
        # sample random directions
        Ndirection = dim
        directions = torch.randn(Ndirection, dim).to(torch.device("cuda:0"))
        directions = directions / torch.sqrt(torch.sum(directions**2, dim=1, keepdim=True))
        # project activations over random directions
        projected_activations_example = torch.einsum('bdn,md->bmn', activations_example, directions)
        projected_activations_generated = torch.einsum('bdn,md->bmn', activations_generated, directions)
        # sort the projections
        sorted_activations_example = torch.sort(projected_activations_example, dim=2)[0]
        sorted_activations_generated = torch.sort(projected_activations_generated, dim=2)[0]
        # L2 over sorted lists
        loss += torch.mean( (sorted_activations_example-sorted_activations_generated)**2 ) 
    return loss, image_optimized

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

min_loss = 10

# LBFGS closure function


i = 0
# optimization loop
with open('results/' + name + '/' +name + ".txt", "w") as f:
    def closure():
        optimizer.zero_grad()
        global min_loss
        loss, image_optimized = slicing_loss(parameters, image_example)

        if loss < min_loss:
            min_loss = loss
            print(i)
            print(loss)
            print(parameters) 
            tmp = image_optimized.detach().cpu().clone().numpy()
            tmp = tmp[0, ...]
            tmp = tmp.transpose(1, 2, 0)
            f.write(str(i) + " - " + str(min_loss.item()) + " : " + str(parameters[0][0].item()) + " " + str(parameters[0][1].item()) + " " + str(parameters[0][2].item()) + " " + 
                    str(parameters[0][3].item()) + " " + str(parameters[0][4].item()) + " " + str(parameters[0][5].item()) + " " + 
                    str(parameters[0][6].item()) + " " + str(parameters[0][7].item()) + " " + str(parameters[0][8].item()) + " " + 
                    str(parameters[0][9].item()) + " " + str(parameters[0][10].item()) + " " + str(parameters[0][11].item()) + " " +
                    str(parameters[0][12].item()) + " " + str(parameters[0][13].item()) + " " + str(parameters[0][14].item()) + " " + str(parameters[0][15].item()) + "\n")
            saveImage('results/' + name + '/pred_' + str(i) + '.png', tmp)

        loss.backward()
        return loss, image_optimized

    for iteration in range(2000):
        loss, image = optimizer.step(closure)
        
        scheduler.step()
        i += 1
