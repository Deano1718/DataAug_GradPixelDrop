from __future__ import print_function
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import datetime
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.hub import load_state_dict_from_url
from torch.utils.model_zoo import load_url as load_state_dict_from_url
#from torch.utils.tensorboard import SummaryWriter
import copy

#from models.wideresnet import *
from models.resnet import *
from utils import *
#from torchinfo import summary

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

import statistics
# import augmentations
# from color_jitter import *
# from diffeomorphism import *
# from rand_filter import *

# from torch.distributions import Dirichlet, Beta
# from einops import rearrange, repeat
# from opt_einsum import contract

#from trades import trades_loss
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import L2DeepFoolAttack, LinfProjectedGradientDescentAttack
#from create_data import compute_smooth_data, merge_data, CustomDataSet

#from robustness.datasets import CustomImageNet
#from robustness.datasets import DATASETS, DataSet, CustomImageNet
#import smoothers


parser = argparse.ArgumentParser(description='PyTorch CIFAR + proximity training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--beta', default=1.0, type=float,
                    help='loss weight for proximity')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many epochs to wait before logging training status')
parser.add_argument('--model-dir', default='../ProtoRuns/kfold-cifar10-',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--model',default="ResNet18",
                    help='network to use')
parser.add_argument('--restart',default=0, type=int,
                    help='restart training, make sure to specify directory')

parser.add_argument('--restart-epoch',default=0, type=int,
                    help='epoch to restart from')

parser.add_argument('--optimizer',default="SGD",
                    help='type of optimizer')

# parser.add_argument('--norm-type', default='batch',
#                     help='batch, layer, or instance')

#parser.add_argument('--par-grad-mult', default=10.0, type=float,
#                    help='boost image gradients if desired')
#parser.add_argument('--par-grad-clip', default=0.01, type=float,
#                    help='max magnitude per update for proto image updates')
#parser.add_argument('--class-centers', default=1, type=int,
#                    help='number of parametric centers per class')
parser.add_argument('--dataset', default="CIFAR10",
                    help='which dataset to use, CIFAR10, CIFAR100, IN100')

parser.add_argument('--anneal', default="stairstep", 
                    help='type of LR schedule stairstep, cosine, or cyclic')
#parser.add_argument('--inter-mix', default=0, type=int,
#                    help='fill in holes within same class')
# parser.add_argument('--augmix', default=0, type=int,
#                     help='use augmix data augmentation')
# parser.add_argument('--prime', default=0, type=int,
#                     help='use PRIME data augmentation')
# parser.add_argument('--confusionmix', default=0, type=int,
#                     help='use confusionmix data augmentation')
# parser.add_argument('--js-loss', default=0, type=int,
#                     help='use jensen shannon divergence for augmix')
#parser.add_argument('--pipeline', nargs='+',default=[],
#                    help='augmentation pipeline')
parser.add_argument('--grad-clip', default = 1, type=int,
                    help='clip model weight gradients by 0.5')
#parser.add_argument('--confusion-mode', default = 2, type=int,
#                    help='0 = (mode0,mode0), 1 = (mode1,mode1), 2= (mode0,mode1) 3= (random,random)')
#parser.add_argument('--mode0rand', default = 0, type=int,
#                    help='randomly switch between window crop size 3 and 5 in mode 0')
parser.add_argument('--channel-norm', default = 0, type=int,
                    help='normalize each channel by training set mean and std')
#parser.add_argument('--channel-swap', default = 0, type=int,
#                    help='randomly permute channels augmentation')
#parser.add_argument('--picid', nargs='+', default=[0,6100,11100,16100,21100,26100,31100,36100],
#                    help='data samples to track')
parser.add_argument('--droprate', default=0.98, type=float,
                    help='droprate for max score')
parser.add_argument('--quantile', default=0.80, type=float,
                    help='initial cutoff quantile')
parser.add_argument('--piclabels', nargs='+', type=int, default=[0,1,2,3,4,5,6,7,8,9],
                    help='labels for pictures')
parser.add_argument('--idstart', default=0, type=int,
                    help='id to start finding target ids')
parser.add_argument('--lr-layer', nargs='+', type=float, default=[0.1,0.1,0.1,0.1],
                    help='lr for each layer')
parser.add_argument('--lr-epoch', default=3, type=int,
                    help='epoch to modify learning rates')
parser.add_argument('--lr-freeze', nargs='+', type=int, default=[0,0,0,0],
                    help='what layers to freeze')
parser.add_argument('--adv', default=0, type=int,
                    help='whether to use adversarial training')
parser.add_argument('--datafrac', default=0.02, type=float,
                    help='fraction of data to use')
parser.add_argument('--epoch-save', nargs='+', type=int, default=[3,5,15,25,35,45,55,65,75,85,95],
                    help='last epoch to save pictures')
parser.add_argument('--par-sparse',type=int, default=0,
                    help='whether to apply l1 sparsity on features')
parser.add_argument('--zeta', type=float, default=0.01,
                    help='weight on l1 sparsity loss')
parser.add_argument('--flip', type=int, default=0,
                    help='whether mask is high or low')
parser.add_argument('--pre-adv', type=int, default=0,
                    help='TRADES loss before masking')
parser.add_argument('--post-adv', type=int, default=0,
                    help='TRADES loss after masking')
parser.add_argument('--post-xent', type=int, default=0,
                    help='xent loss after masking')
parser.add_argument('--natmask', type=int, default=1,
                    help='use the masked image as the reference for TRADES')
                    






#lyrs=[4,4,4], plns=[64,128,256], cutoffs=[0,0,0], dirs=[0,0,0], reds=[1,1,0], cat=0, resFinish=1,





# AugMix options
#parser.add_argument(
#    '--mixture-width',
#    default=3,
#    type=int,
#    help='Number of augmentation chains to mix per augmented example')
#parser.add_argument(
#    '--mixture-depth',
#    default=-1,
#    type=int,
#    help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
#parser.add_argument(
#    '--aug-severity',
#    default=3,
#    type=int,
#    help='Severity of base augmentation operators')
# parser.add_argument(
#     '--no-jsd',
#     '-nj',
#     action='store_true',
#     help='Turn off JSD consistency loss.')
#parser.add_argument(
#    '--all-ops',
#    default=1,
#    type=int,
#    help='Turn on all operations (+brightness,contrast,color,sharpness).')



args = parser.parse_args()

kwargsUser = {}

#kwargsUser['norm_type'] = args.norm_type
#kwargsUser['augmix'] = args.augmix
#kwargsUser['prime'] = args.prime
#kwargsUser['js_loss'] = args.js_loss

  
# settings
if (args.model == "ResNet18"):
    network_string = "ResNet18"
elif (args.model == "TrilB"):
    network_string = "TrilB"
elif (args.model == "TrilC"):
    network_string = "TrilC"
elif (args.model == "TrilD"):
    network_string = "TrilD"
else:
    print ("Invalid model architecture")
    
def get_datetime():
    now = datetime.now()
    dt_string = now.strftime("%m%d_%H_%M_%S")
    return dt_string



#model_dir = ("{}_{}_beta_{}_k_{}_pool_{}_norm_{}_{}".format("../ProtoRuns/metric-{}".format(args.dataset),network_string,args.beta,args.k,args.proto_pool, args.proto_norm,get_datetime()))


targ = "/home/lab/nxd551/Desktop/GradPixRuns"
pictarg = "/home/lab/nxd551/Desktop/GradPixelDrop/pictures"
date_time = get_datetime()
#plottarg = "/home/lab/csc-232/nxd551-210/Desktop/PrototypeImage/metric_plots"
dir_suffix = "gp-{}_{}_{}".format(args.dataset,network_string,get_datetime())
#need local global results file                                                                                                                                                                             
#local_results_name = "../ProtoRuns/quicklook_{}_{}.txt".format(args.dataset,get_datetime())                                                                                                                        

#full_dir = os.path.join(targ, model_dir)
model_dir = os.path.join(targ, dir_suffix)
#full_dir_plot = os.path.join(plottarg,dir_suffix)

#print (full_dir)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(pictarg):
    os.makedirs(pictarg)


# if not os.path.exists(full_dir_plot):
#     os.makedirs(full_dir_plot)


with open('{}/commandline_args.txt'.format(model_dir), 'a') as f:
    json.dump(args.__dict__, f, indent=2)
f.close()

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': False} if use_cuda else {}
torch.cuda.empty_cache()

print ("cuda: ", use_cuda)
kwargsUser['deviceModel'] = device
print (args.optimizer)




# def vis_grad(img):
#     std = 3 * np.std(img)
#     img = np.clip(img, a_max=std, a_min=-std)
#     img = (1 + img / std) * 0.5
#     plt.imshow(img)

def sort_list(list1, list2):
 
    zipped_pairs = zip(list2, list1)
 
    z = [x for _, x in sorted(zipped_pairs)]
 
    return z

def trades_loss(model,
                x_natural,
                x_mask,
                y,
                optims,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf',
                transformDict={}):
    # define KL-loss
    #criterion_kl = nn.KLDivLoss(size_average=False)
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_mask.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            x_adv_norm = transformDict['norm'](x_adv)
            x_nat_norm = transformDict['norm'](x_natural)
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv_norm), dim=1),
                                       F.softmax(model(x_nat_norm), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    

    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    #optimizer.zero_grad()
    #for o in optims:
    #    o.zero_grad()

    x_adv_norm = transformDict['norm'](x_adv)
    x_nat_norm = transformDict['norm'](x_natural)
    # calculate robust loss
    logits = model(x_nat_norm)
    #loss_natural = F.cross_entropy(logits, y)
    loss_robust = criterion_kl(F.log_softmax(model(x_adv_norm), dim=1),
                                                    F.softmax(logits, dim=1))
    #loss = loss_natural + beta * loss_robust
    return loss_robust


def generic_attack(model, X, Ytarg, ini_pert, iterations, step_size, lim=4.0/255.0, limType='inf', targeted=0, frozen=0, transformDict={}):
    
    A, B, C, D = X.size()
    #source_labels = torch.arange(A, dtype=torch.long, device=device)
    model.eval()
    model.multi_out = 0
    for p in model.parameters():
        p.requires_grad = False


    #last_mat_tracker = torch.zeros_like(input_img)
    #last_projmat_tracker = torch.ones_like(input_img)
    #baseStep_matrix = baseStep*torch.ones_like(input_img)

    X_orig = X.clone().detach()

    global_noise_data = ini_pert*torch.rand([A,B,C,D],dtype=torch.float, device=device)
    #image_cache_adv = X_orig.clone().detach()
    #image_cache_src = X_orig.clone().detach()

    #mask_frozen = torch.zeros(A, dtype=torch.bool, device=device)

    for i in range(iterations):

        noise_batch = global_noise_data[0:A].clone().detach().requires_grad_(True).to(device)

        _inputs = X_orig + noise_batch
        _inputs.clamp_(0.0, 1.0)

        _inputs_norm = transformDict['norm'](_inputs)
        
        #no output for stylization "network"
        Z = model(_inputs_norm)

        loss = F.cross_entropy(Z, Ytarg)

        loss.backward()

        with torch.no_grad():

            pert_vector = torch.sign(noise_batch.grad)
            #grad_mag = gradients_unscaled.view(gradients_unscaled.shape[0], -1).norm(2, dim=-1)
            #gradients = baseStep*gradients_unscaled  / grad_mag.view(-1, 1, 1, 1)

            #update perturbation field
            if targeted:
                global_noise_data[0:A] -= step_size*pert_vector.data
            else:
                global_noise_data[0:A] += step_size*pert_vector.data

            #project noise if necessary
            if limType =='inf':
                global_noise_data[0:A].clamp_(-lim,lim)
            elif limType == 'l2':
                mags = torch.linalg.norm(global_noise_data[0:A].view(A,-1), dim=1)
                mask = mags > lim      #should be shape [A]

                #print ("mask shape", mask.shape)

                scaling_factors = mags[mask] / lim   #should be shape [A]

                #print ("scaling factor shape", scaling_factors.shape)
                #print ("global noise[mask] shape ", global_noise_data[mask].shape)

                global_noise_data[mask] /= scaling_factors.view(-1,1,1,1)

            noise_batch.grad.zero_()


    adv_imgs = (X_orig + global_noise_data[0:A].clone().detach()).clone().detach()
    adv_imgs.clamp_(0.0,1.0)

    model.train()

    return adv_imgs.clone().detach()

def train(args, model, device, cur_loader, optims, epoch, scheduler=0.0, max_steps = 0, idlist=[], lab_to_id={}, epoch_list=[], adv=0, transformDict={}, **kwargs):

    model.train()
    model.multi_out=0

    #check_tensor = torch.tensor(args.picid).to(device)
    check_tensor = torch.tensor(idlist).to(device)

    temp_pic_id = []
    temp_pic_grad_list = []
    temp_pic_mask_list = []

    if epoch in epoch_list:
        for ct in check_tensor:
            temp_pic_id.append(0.0)
            temp_pic_grad_list.append(0.0)
            temp_pic_mask_list.append(0.0)

    #if epoch > 10 and epoch < 50:
    #    cur_loader.dataset.dropscores(args.droprate)


    print ('Training model')

    with torch.no_grad():
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)                                                              
        convSum = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3, device=device)
        conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3, device=device)
        conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3, device=device)
        conv3 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3, device=device)
        conv4 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3, device=device)

        sum_filt = torch.ones((1,3,3), device=device)
        conv1_filt = -0.25*torch.ones((1,3,3), device=device)
        conv1_filt[0,1,1]=2.0

        conv2_filt = -1.0*torch.ones((1,3,3), device=device)
        conv2_filt[0,1,1]=0.0
        conv2_filt[0,0,1] = 1.0
        conv2_filt[0,1,0] = 1.0
        conv2_filt[0,1,2] = 1.0
        conv2_filt[0,2,1] = 1.0

        conv3_filt = (-1.0*conv2_filt).clone()
        conv4_filt = (-1.0*conv1_filt).clone()

        sumFilt=[]
        conv1Filt=[]
        conv2Filt=[]
        conv3Filt=[]
        conv4Filt=[]

        for _ in range(3):
            sumFilt.append(sum_filt.clone())
            conv1Filt.append(conv1_filt.clone())
            conv2Filt.append(conv2_filt.clone())
            conv3Filt.append(conv3_filt.clone())
            conv4Filt.append(conv4_filt.clone())

        sumFilt = nn.Parameter(torch.stack(sumFilt,dim=0).clone(),requires_grad=False)
        conv1Filt = nn.Parameter(torch.stack(conv1Filt,dim=0).clone(), requires_grad=False)
        conv2Filt = nn.Parameter(torch.stack(conv2Filt,dim=0).clone(), requires_grad=False)
        conv3Filt = nn.Parameter(torch.stack(conv3Filt,dim=0).clone(), requires_grad=False)
        conv4Filt = nn.Parameter(torch.stack(conv4Filt,dim=0).clone(), requires_grad=False)

        convSum.weight = sumFilt   #need 32,1,3,3
        conv1.weight = conv1Filt
        conv2.weight = conv2Filt
        conv3.weight = conv3Filt
        conv4.weight = conv4Filt

    ave_score = []
    ave_score1 = []
    ave_score2 = []
    ave_score3 = []
    ave_score4 = []
    #weight=torch.tensor(0.0)
    for batch_idx, (data, target, scores, inds) in enumerate(cur_loader):

        data, target, scores, inds = data.to(device), target.to(device), scores.to(device), inds.to(device)

        A,B,C,D = data.size()
        #print (batch_idx)
        #print (torch.min(data))
        #print ("mean ", MEAN)

        #optimizer.zero_grad()
        for o in optims:
            o.zero_grad()

        loss= 0.0

        if adv and args.pre_adv:
            # adversaries= generic_attack(model,
            #                             data.clone().detach(),
            #                             target,
            #                             ini_pert=0.1,
            #                             iterations=7,
            #                             step_size=1.0/255,
            #                             lim=4.0/255,
            #                             limType='inf',
            #                             targeted=0,
            #                             frozen=0,
            #                             transformDict=transformDict)

            # adv_norm = transformDict['norm'](adversaries)
            # Zadv = model(adv_norm)
            # loss += args.beta*F.cross_entropy(Zadv, target)

            loss += args.beta*trades_loss(model,
                x_natural=data.clone(),
                y=target,
                optims=optims,
                step_size=0.01,
                epsilon=0.031,
                perturb_steps=7,
                beta=1.0,
                distance='l_inf',
                transformDict=transformDict)


        model.multi_out=1
        
        data_fwd = data.clone().detach().requires_grad_(True)

        #channel normalize data
        
        data_fwd_norm = transformDict['norm'](data_fwd)

        #print (weight)

        ftr, Z = model(data_fwd_norm)
        #print (Z)
        loss += F.cross_entropy(Z, target)

        if args.par_sparse:
            #l2 = torch.linalg.norm(_par_images_opt.view(10,-1),ord=2,dim=1)
            l1 = torch.linalg.norm(ftr.view(ftr.shape[0],-1),ord=1,dim=1)
            loss += args.zeta*torch.mean(l1)


        #if adv and args.post_adv:
        #    loss.backward(retain_graph=True)
        #else:
        loss.backward()

        #compute modified data
        with torch.no_grad():
            gradients_unscaled = data_fwd.grad.clone().detach()
            #_par_images_opt.grad.zero_()
            data_fwd.grad.zero_()
            std, mean = torch.std_mean(gradients_unscaled.view(A,-1),dim=1)

            #max_lim = mean + 3.0*(std)
            #min_lim = mean - 3.0*(std)

            max_lim = 3.0*std
            min_lim = -3.0*std

            gradients_unscaled = torch.max(torch.min(gradients_unscaled, max_lim.view(-1,1,1,1)), min_lim.view(-1,1,1,1))
            #print ("grad unscaled shape ", gradients_unscaled.shape)
            #print (torch.max(torch.abs(gradients_unscaled)))
            #gradients_unscaled.clamp_(-0.031,0.031)
            grad_mag = gradients_unscaled.view(gradients_unscaled.shape[0], -1).norm(2, dim=-1)
            gradients = gradients_unscaled  / grad_mag.view(-1, 1, 1, 1)
            img = (1.0 + gradients_unscaled / max_lim.view(-1,1,1,1)) * 0.5
            
            #raw_grads = data_fwd.grad.clone()
            #abs_grads = torch.abs(raw_grads)
            #sign_grads = torch.sign(raw_grads)
            if epoch in epoch_list:
                #img = (1.0 + gradients_unscaled / max_lim.view(-1,1,1,1)) * 0.5
                #print ("img shape",img.shape)
                for val in check_tensor:
                    if torch.any(inds==val):
                        #temp_pic_id.append(val)
                        for targ in target[inds==val]:
                            #print (targ)
                            temp_pic_grad_list[lab_to_id[targ.clone().cpu().item()]] = img[inds==val].clone().squeeze().cpu()
                        #temp_pic_grad_list.append(img[inds==val].clone().squeeze().cpu())

            #channel_sums = convSum(abs_grads)    #still [B,3,H,W]
            #norm_grads = raw_grads / (channel_sums + 0.001)
            
            #channel_scores = conv1(norm_grads)
            abs_grads = torch.abs(gradients)

            channel_scores_1 = torch.abs(conv1(abs_grads))   #A,3,32,32
            channel_scores_2 = torch.abs(conv2(abs_grads))
            channel_scores_3 = torch.abs(conv3(abs_grads))
            channel_scores_4 = torch.abs(conv4(abs_grads))

            total_scores_1 = torch.sum(channel_scores_1, dim=1, keepdim=True)  #A,1,32,32
            total_scores_2 = torch.sum(channel_scores_2, dim=1, keepdim=True)
            total_scores_3 = torch.sum(channel_scores_3, dim=1, keepdim=True)
            total_scores_4 = torch.sum(channel_scores_4, dim=1, keepdim=True)

            #A,4,32,32 --> A,1,32,32
            #final_score_tensor = torch.max(torch.cat((total_scores_1,total_scores_2,total_scores_3,total_scores_4),dim=1), dim=1, keepdim=True)   #A,1,32,32  max score at each element
            score_tensor1 = torch.maximum(total_scores_1,total_scores_4)
            score_tensor2 = torch.maximum(total_scores_3,total_scores_2)
            final_score_tensor = torch.maximum(score_tensor1,score_tensor2)
            
            #final_score_batch = torch.mean(final_score_tensor.view(A,-1), dim=1)
            ave_score.append(torch.mean(final_score_tensor).item())
            ave_score1.append(torch.mean(total_scores_1).item())
            ave_score2.append(torch.mean(total_scores_2).item())
            ave_score3.append(torch.mean(total_scores_3).item())
            ave_score4.append(torch.mean(total_scores_4).item())
            #print (final_score.shape)

            #moving average every 5 epochs
            if epoch > 5 and epoch < 100:
                cutoff_scores = torch.quantile(final_score_tensor.view(A,-1),args.quantile,dim=1)
                #print ("cutoff shape ", cutoff_scores.shape)
                cur_loader.dataset.contribute_average(inds, cutoff_scores.clone())
            
            if (epoch % 5)==0:
                #update cutoff score
                cur_loader.dataset.transfer_ave()
                cur_loader.dataset.reset_average()
        
            #total_scores = torch.sum(channel_scores, dim=1, keepdim=True)    #
            #total_scores_norm = F.instance_norm(total_scores)
            #print (total_scores_norm[0])
            #print (total_scores_norm[1])
            #print (total_scores_norm.shape)

            if args.flip:
                mask = (final_score_tensor > scores.view(A,1,1,1)).expand(-1,3,-1,-1).float()
            else:
                mask = (final_score_tensor < scores.view(A,1,1,1)).expand(-1,3,-1,-1).float()

            mask_filt = Median_Filter(mask, kernel=2)
            #mask_flip = ~mask
            #set_one = torch.logical_and(mask_flip,sign_grads)
            #print ("mask shape ",mask.shape)
                

            #kept_pct = torch.sum(mask) / args.batch_size
            #print (kept_pct)

            #need to set to either 0 or 1 depending on sign of grad
            with torch.no_grad():
                data_fwd_masked = mask_filt*(data.clone().detach())    #set all masked off to zero initially
                #data_fwd_masked[set_one] = 1.0                    #set to one those grads that were noisy and positive
                if epoch in epoch_list:
                    #img = (1.0 + gradients_unscaled / max_lim) * 0.5
                    for val in check_tensor:
                        if torch.any(inds==val):
                            for targ in target[inds==val]:
                                temp_pic_mask_list[lab_to_id[targ.clone().cpu().item()]]= data_fwd_masked[inds==val].clone().squeeze().cpu()
                            #temp_pic_id.append(val)
                            #temp_pic_mask_list.append(data_fwd_masked[inds==val].clone().squeeze().cpu())

        if adv and args.post_adv:
            # adversaries= generic_attack(model,
            #                             data.clone().detach(),
            #                             target,
            #                             ini_pert=0.1,
            #                             iterations=7,
            #                             step_size=1.0/255,
            #                             lim=4.0/255,
            #                             limType='inf',
            #                             targeted=0,
            #                             frozen=0,
            #                             transformDict=transformDict)

            # adv_norm = transformDict['norm'](adversaries)
            # Zadv = model(adv_norm)
            # loss += args.beta*F.cross_entropy(Zadv, target)

            if args.natmask:

                loss += args.beta*trades_loss(model,
                    x_natural=data_fwd_masked.clone(),
                    x_mask=data_fwd_masked.clone(),
                    y=target,
                    optims=optims,
                    step_size=0.01,
                    epsilon=0.031,
                    perturb_steps=7,
                    beta=1.0,
                    distance='l_inf',
                    transformDict=transformDict)

            else:
                loss += args.beta*trades_loss(model,
                    x_natural=data.clone(),
                    x_mask=data_fwd_masked.clone(),
                    y=target,
                    optims=optims,
                    step_size=0.01,
                    epsilon=0.031,
                    perturb_steps=7,
                    beta=1.0,
                    distance='l_inf',
                    transformDict=transformDict)


        if args.lr_epoch and args.post_xent:
            
            data_fwd_masked_norm = transformDict['norm'](data_fwd_masked)
            #data_fwd_masked_norm = transformDict['norm'](img.clone())

            ftr_mask, Z_mask = model(data_fwd_masked_norm)


            loss_mask = F.cross_entropy(Z_mask, target)

            loss_mask.backward()


        if (args.grad_clip):
            nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)


        #optimizer.step()
        for o in optims:
            o.step()


        if args.anneal == "cyclic" or args.anneal == "cosine":
            if batch_idx < max_steps:
                scheduler.step()
                
        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(cur_loader.dataset),
                       100. * batch_idx / len(cur_loader), loss.item()))


    #if temp_pic_grad_list:
    #    new_x1, temp_pic_grad_list = zip(*sorted(zip(temp_pic_id, temp_pic_grad_list)))
    #if temp_pic_mask_list:
    #    new_x2, temp_pic_mask_list = zip(*sorted(zip(temp_pic_id, temp_pic_mask_list)))

    #print (temp_pic_grad_list)
    #print (ave_score)
    return statistics.mean(ave_score), statistics.mean(ave_score1), statistics.mean(ave_score2), statistics.mean(ave_score3), statistics.mean(ave_score4), temp_pic_grad_list, temp_pic_mask_list



def eval_train(model, device, train_loader, transformDict):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            data = transformDict['norm'](data)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader, transformDict):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = transformDict['norm'](data)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(args, optimizers, epoch):
    """decrease the learning rate"""
    #lr = args.lr*(0.99**epoch)
    #print (lr)
    lr=args.lr
    if epoch >= (0.5*args.epochs):
        lr = args.lr * 0.1
    if epoch >= (0.75*args.epochs):
        lr = args.lr * 0.01
    if epoch >= (0.9*args.epochs):
        lr = args.lr * 0.001

    for o, opti in enumerate(optimizers):
        if (epoch>=args.lr_epoch and args.lr_freeze[o]):
            for param_group in opti.param_groups:
                param_group['lr'] = 0.001
        else:
            for param_group in opti.param_groups:
                param_group['lr'] = lr
                
                


def main():
#    # setup data loader
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #MEAN = [0.5]*3
    #STD = [0.5]*3

    MEAN = [0.0]*3
    STD = [1.0]*3
    
    if (args.dataset == "CIFAR10"):
        if args.channel_norm:
            MEAN = [0.4914, 0.4822, 0.4465]
            STD = [0.2471, 0.2435, 0.2616] 
    elif(args.dataset == "CIFAR100"):
        if args.channel_norm:
            MEAN = [0.5071, 0.4865, 0.4409]
            STD = [0.2673, 0.2564, 0.2762]
    elif  (args.dataset == "IN100"):
        if args.channel_norm:
            MEAN = [0.485, 0.456, 0.406]
            STD  = [0.229, 0.224, 0.225]

    elif (args.dataset == "TINYIN"):
        if args.channel_norm:
            MEAN = [0.4802, 0.4481, 0.3975]
            STD  = [0.2302, 0.2265, 0.2262]

    elif (args.dataset == "FASHION"):
        MEAN = [0.5]
        STD = [0.5]
    else:
        print ("ERROR dataset not found")

    gen_transform_train = transforms.Compose([transforms.ToTensor()])
    #gen_transform_test = transforms.Compose([transforms.ToTensor()])

    #first augmentation in pipeline gets [Tensor, Flip, Crop] by default
    if args.dataset in ["CIFAR10","CIFAR100"]:
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4)])

        train_transform_tensor = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4)])
        gen_transform_test = transforms.Compose(
            [transforms.ToTensor()])

    elif args.dataset in ["FASHION"]:

        train_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Pad(2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4)])

        train_transform_tensor = transforms.Compose(
            [transforms.Pad(2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4)])

        gen_transform_test = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Pad(2)])
        
    elif args.dataset in ["TINYIN"]:
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4)])

        train_transform_tensor = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4)])
        gen_transform_test = transforms.Compose(
            [transforms.ToTensor()])

    elif args.dataset in ["IN100"]:
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip()])

        train_transform_tensor = transforms.Compose(
            [transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip()])
        gen_transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize(256),
             transforms.CenterCrop(224)])
    else:
        print ("ERROR setting transforms")


    #comp_list_test = [transforms.ToTensor()]
    
    if (args.dataset == "CIFAR10"):

        trainset_basic = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=gen_transform_test)
        train_loader_basic = torch.utils.data.DataLoader(trainset_basic, batch_size=args.batch_size, shuffle=True, **kwargs)

        #both augmix and PRIME want [crop, flip] before their augmentations
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=train_transform)
        #train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)


        #cur_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)


        testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=gen_transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        #num_classes = 10
        kwargsUser['num_classes'] = 10
        nclass=10
        nchannels = 3
        H, W = 32, 32
            
    elif (args.dataset == "CIFAR100"):

        trainset_basic = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=gen_transform_test)
        train_loader_basic = torch.utils.data.DataLoader(trainset_basic, batch_size=args.batch_size, shuffle=True, **kwargs)


        #trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)
        trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)
        #train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        #cur_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=gen_transform_test)
        #test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        #num_classes = 100
        kwargsUser['num_classes'] = 100
        nclass=100
        nchannels = 3
        H, W = 32, 32

    elif (args.dataset == "FASHION"):

        #trainset_basic = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=gen_transform_train)
        #train_loader_basic = torch.utils.data.DataLoader(trainset_basic, batch_size=args.batch_size, shuffle=True, **kwargs)


        #trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)
        trainset = torchvision.datasets.FashionMNIST(root='../data', train=True, download=True, transform=train_transform)
        #train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        #cur_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        testset = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True, transform=gen_transform_test)
        #test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        #num_classes = 100
        kwargsUser['num_classes'] = 10
        nclass=10
        nchannels = 1
        H, W = 32, 32

    elif (args.dataset == "IN100"):
        
        #trainset_basic = datasets.ImageFolder(
        #    './Data_ImageNet/train_100',
        #    transform=gen_transform_train)
        #train_loader_basic = torch.utils.data.DataLoader(trainset_basic, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)


        trainset = datasets.ImageFolder(
            './Data_ImageNet/train_100',
            transform=train_transform)
        #train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        #cur_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        testset = datasets.ImageFolder(
            './Data_ImageNet/val_100',
            transform=gen_transform_test)
        #test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        kwargsUser['num_classes'] = 100
        nclass = 100
        nchannels = 3
        H, W = 224, 224
    
    elif (args.dataset == "TINYIN"):
        
        #trainset_basic = datasets.ImageFolder(
        #    './Data_ImageNet/train_100',
        #    transform=gen_transform_train)
        #train_loader_basic = torch.utils.data.DataLoader(trainset_basic, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)


        trainset = datasets.ImageFolder(
            './tiny-imagenet-200/train',
            transform=train_transform)
        #train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        #cur_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        testset = datasets.ImageFolder(
            './tiny-imagenet-200/val/images',
            transform=gen_transform_test)
        #test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        kwargsUser['num_classes'] = 200
        nclass = 200
        nchannels = 3
        H, W = 64, 64

    else:
          
        print ("Error getting dataset")


    transformDict = {}

    transformDict['basic'] = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(H, padding=4),transforms.Normalize(MEAN, STD)])
    transformDict['flipcrop'] = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(H, padding=4)])
    transformDict['norm'] = transforms.Compose([transforms.Normalize(MEAN, STD)])
    transformDict['mean'] = MEAN
    transformDict['std'] = STD


    with open('{}/train_hist.txt'.format(model_dir), 'a') as f:
        f.write("\n")
    f.close()


    all_inds = np.arange(len(trainset_basic))
    inds_tr, inds_te, y_tr, y_te = train_test_split(all_inds, trainset.targets, test_size=0.01, random_state=args.seed, stratify=trainset.targets)   #.02*5000 = 100 samples per class
    #inds_tr, inds_te, y_tr, y_te = train_test_split(all_inds, trainset.targets, test_size=d, random_state=args.seed, stratify=trainset.targets)

    subtrain0 = torch.utils.data.Subset(trainset_basic, inds_te)
    scored_dataset = CustomDataSetScore(trainset_basic, transformDict['flipcrop'])


    cur_loader = torch.utils.data.DataLoader(scored_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    adv_loader = torch.utils.data.DataLoader(subtrain0, batch_size=100, shuffle=False, **kwargs)
    

    # init model, ResNet18() can be also used here for training
    if (args.model == "ResNet18"):
        model = ResNet18(nclass = nclass, scale=1.0, channels=nchannels, **kwargsUser).to(device)
    elif (args.model == "TrilB"):
        model = TrilNet_B(nclass = nclass, channels=nchannels, **kwargsUser).to(device)
    elif (args.model == "TrilC"):
        model = TrilNet_C(nclass = nclass, channels=nchannels, **kwargsUser).to(device)
    elif (args.model == "TrilD"):
        model = TrilNet_D(lyrs=args.layers, plns=args.planes, cutoffs=args.cutoffs, dirs=args.dirs, reds=args.reds, cat=args.cat, resFinish=args.resFinish, 
                          nclass=10, channels=3, **kwargsUser).to(device)
    else:
        print ("INVALID MODEL ARCHITECTURE")

    #print (summary(model, input_size=(1, 3, 32, 32)))

    model.train()


    if args.anneal in ["stairstep", "cosine"]:
        lr_i = args.lr
    elif args.anneal in ["cyclic"]:
        lr_i = 0.2
    else:
        print ("Error setting learning rate")

    print (lr_i)
    print (args.lr_layer)
    all_opts = []
    if args.optimizer == 'SGD':
        #optimizer = optim.SGD(model.parameters(), lr=lr_i, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        opt1 = optim.SGD([*model.conv1.parameters(), *model.layer1.parameters()], lr=args.lr_layer[0], momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        opt2 = optim.SGD([*model.layer2.parameters(), *model.layer3.parameters()], lr=args.lr_layer[1], momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        opt3 = optim.SGD([*model.layer4.parameters()], lr=args.lr_layer[2], momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        opt4 = optim.SGD([*model.linear.parameters()], lr=args.lr_layer[3], momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

        all_opts.append(opt1)
        all_opts.append(opt2)
        all_opts.append(opt3)
        all_opts.append(opt4)


    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr_i)
    
    scheduler = 0.0
    steps_per_epoch = int(np.ceil(len(cur_loader.dataset) / args.batch_size))

    print ("len(cur_loader.dataset)", len(cur_loader.dataset))
    print ("len(cur_loader)", len(cur_loader))

    if args.anneal == "stairstep":
        pass
    elif args.anneal == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*len(cur_loader), eta_min=0.0000001, last_epoch=-1, verbose=False)
    elif args.anneal == "cyclic":
        pct_start = 0.25
        #steps_per_epoch = 391   #50k / 128
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr_i, epochs = args.epochs, steps_per_epoch = steps_per_epoch, pct_start = pct_start)
    else:
        print ("ERROR making scheduler") 


    
    piclist = []
    id_list = []
    #label_list = [n for n in range(nclass)]
    label_to_id_dict = {}
    label_list = []

    for p, lb in enumerate(args.piclabels):
        label_list.append(lb)
        label_to_id_dict[lb] = p
        piclist.append(0.0)

    #label_list = [a for a in args.piclabels]

    print (label_list)
    #for lb in label_list:
    #    piclist.append(0.0)
        
    #for i in args.picid:
    for i in range(args.idstart, len(scored_dataset)):
        #x_pic, y_pi = trainset_basic[i]
        x_pic, y_pi = scored_dataset.dataset[i]
        if y_pi in label_list:
            print (y_pi)
            label_list.pop(label_list.index(y_pi))
            #piclist[y_pi] = x_pic.clone()
            piclist[label_to_id_dict[y_pi]] = x_pic.clone()
            id_list.append(i)   
        #print (x_pic.shape)
        #piclist.append(x_pic.clone())
    print (args.piclabels)

    current_adv = 0

    #save_epoch_list = range(args.epoch_save)
    save_epoch_list = args.epoch_save

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        if args.anneal == "stairstep":
            adjust_learning_rate(args, all_opts, epoch)

        if epoch == args.lr_epoch and args.adv:
            current_adv=1

        #with open('{}/lr_hist.txt'.format(model_dir), 'a') as f:lab_to_id[targ.clone().cpu().item()]
        #    f.write("{0:3.6f}".format(scheduler.get_last_lr()[0]))
        #    f.write("\n")
        #f.close()

        #print (scheduler.get_last_lr()[0])
        # proximity training
        mean_score, mean_score1, mean_score2, mean_score3, mean_score4, pic_grad_list, pic_mask_list = train(args, 
            model, 
            device, 
            cur_loader, 
            all_opts, 
            epoch, 
            scheduler=scheduler, 
            max_steps = steps_per_epoch, 
            idlist=id_list,
            lab_to_id=label_to_id_dict,
            epoch_list=save_epoch_list,
            adv=current_adv, 
            transformDict=transformDict, 
            **kwargsUser)

        #train(args, model, device, cur_loader, optimizer, epoch, scheduler=0.0, max_steps = 0, passes=5, transformDict={}, **kwargs):
        piclist.extend(pic_grad_list)
        piclist.extend(pic_mask_list)

        # evaluation on natural examples
        model.eval()
        model.multi_out=0
        print('================================================================')
        loss_train, acc_train = eval_train(model, device, train_loader_basic, transformDict)
        loss_test, acc_test = eval_test(model, device, test_loader, transformDict)
        print('================================================================')

        adv_acc = 0.0

        if adv:
            model.eval()
            model.multi_out=0
            attack = LinfProjectedGradientDescentAttack(abs_stepsize=0.01, steps=7)
            preprocessing = dict(mean=MEAN, std=STD, axis=-3)
            fmodel = PyTorchModel(model, bounds=(0,1), preprocessing=preprocessing)

            correct = 0.0

            for batch_idx, (data, target) in enumerate(adv_loader):
   
                raw, X_new_torch, is_adv = attack(fmodel, data.clone(), target.clone(), epsilons=0.031)

                data_adv = transformDict['norm'](X_new_torch)
                output = model(data_adv)

                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
            
            adv_acc = correct / len(adv_loader.dataset)

        with open('{}/train_hist.txt'.format(model_dir), 'a') as f:
            f.write("{0:4.3f}\t{1:4.3f}{2:4.3f}\t{3:4.4f}\t{4:4.4f}\t{5:4.4f}\t{6:4.4f}\t{7:4.4f}\t{8:4.4f}\t{9:4.4f}\n".format(acc_train,acc_test,adv_acc,loss_train,loss_test, mean_score, mean_score1, mean_score2, mean_score3, mean_score4))
        f.close()

        if (epoch == args.epochs):
            torch.save(model.state_dict(),os.path.join(model_dir, 'model-{}-epoch{}.pt'.format(network_string,epoch)))


    #print (piclist[0].shape)
    for t in piclist:
        t = t.squeeze(0)

    max_pics_per_set = 6*len(args.piclabels)
    #epochs_per_set = args.epoch_save//(len(piclist)//len(args.piclabels))
    #epochs_per_set = args.epoch_save//len(args.piclabels)
    epochs_per_set = 3

    print ("len piclist",len(piclist))
    #print (pics_per_set)
    #print (epochs_per_set)

    jmax = max(1, len(piclist)//(max_pics_per_set))

    if jmax == 1:
        max_pics_per_set = len(piclist)
    
    for j in range(jmax):
    
        piclist_cur = torch.stack(piclist[j*max_pics_per_set:(j+1)*max_pics_per_set], dim=0)
        print (piclist_cur.shape)
        img = torchvision.utils.make_grid(piclist_cur.clone().cpu(), nrow=len(args.piclabels))
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

        plt.savefig("{}/Images_lr{}_ep{}freeze{}{}{}{}_classes{}{}_adv{}_ep{}{}_{}{}flip{}_{}.png".format(pictarg, 
                                                                                               args.lr_layer[0],
                                                                                               args.lr_epoch, 
                                                                                               args.lr_freeze[0],
                                                                                               args.lr_freeze[1],
                                                                                               args.lr_freeze[2],
                                                                                               args.lr_freeze[3], 
                                                                                               args.piclabels[0], 
                                                                                               args.piclabels[-1], 
                                                                                               args.adv,
                                                                                               j*epochs_per_set,
                                                                                               (j+1)*epochs_per_set,
                                                                                                    args.droprate,
                                                                                                    args.quantile,
                                                                                                    args.flip,
                                                                                               date_time))

        plt.close()



if __name__ == '__main__':
    main()

