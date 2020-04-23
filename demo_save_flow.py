import time
import os
from torch.autograd import Variable
import math
import torch
import torch.nn.functional as F


import random
import numpy as np
import numpy
import networks
from my_args import  args

from scipy.misc import imread, imsave, imresize
import cv2

from AverageMeter import  *

import gc

torch.backends.cudnn.benchmark = True # to speed up the


DATA_PATH = '/data'
OUTPUT_PATH = '/results'


if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)



model = networks.__dict__[args.netName](channel=args.channels,
                            filter_size = args.filter_size ,
                            timestep=args.time_step,
                            training=False)


def optical_flow_warp(img, optical_flow):
    grid = optical_flow.permute(0, 2, 3, 1)
    projected_img = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros")
    return projected_img

def image_translated_with_flow_torch(rgb_1,flow):
    u = flow[:,:,0]
    v = flow[:,:,1]
    u = u * 2. / (flow.shape[0]/2)
    v = v * 2. / (flow.shape[0]/2)
    
    u = np.expand_dims(u, axis = 0)
    v = np.expand_dims(v, axis = 0)
    
    flow_stacked = np.stack((u,v), axis=1)
    
    flow_stacked_neutral = np.meshgrid(np.linspace(-1,1,flow.shape[0]), np.linspace(-1,1,flow.shape[1]))
    flow_stacked_neutral = np.expand_dims(np.array(flow_stacked_neutral), axis=0)
    
    flow_torch = torch.FloatTensor(flow_stacked_neutral-flow_stacked)
    
    rgb_torch = torch.FloatTensor(np.expand_dims(rgb_1, axis=0))
    rgb_torch = rgb_torch.permute(0,3,1,2)
    
    projected_img = optical_flow_warp(rgb_torch, flow_torch)
    
    out_image = projected_img.permute(0,2,3,1).numpy()
    out_image = np.squeeze(out_image)
    
    return out_image



def image_translated_with_flow(rgb_1,flow):
    rgb_2_from_rgb_1_with_flow = np.zeros_like(rgb_1)
    for x in range(rgb_1.shape[0]):
        for y in range(rgb_1.shape[1]):
            displacement_x = flow[x, y, 0]*2
            displacement_y = flow[x, y, 1]*2
            try:
                #print(x+displacement_x, y-displacement_y)
                if int(x-displacement_y) > 0 and int(y+displacement_x) > 0:
                    rgb_2_from_rgb_1_with_flow[int(x+displacement_y), int(y+displacement_x)] = rgb_1[x,y]
            except:
                pass
    return rgb_2_from_rgb_1_with_flow



if args.use_cuda:
    model = model.cuda()

args.SAVED_MODEL = './model_weights/best.pth'
if os.path.exists(args.SAVED_MODEL):
    print("The testing model weight is: " + args.SAVED_MODEL)
    if not args.use_cuda:
        pretrained_dict = torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage)
        # model.load_state_dict(torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage))
    else:
        pretrained_dict = torch.load(args.SAVED_MODEL)
        # model.load_state_dict(torch.load(args.SAVED_MODEL))

    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    # 4. release the pretrained dict for saving memory
    pretrained_dict = []
else:
    print("*****************************************************************")
    print("**** We don't load any trained weights **************************")
    print("*****************************************************************")
torch.set_grad_enabled(False)

model = model.eval() # deploy mode

torch.set_grad_enabled(False)

use_cuda=args.use_cuda
save_which=args.save_which
dtype = args.dtype
unique_id =str(random.randint(0, 100000))
print("The unique id for current testing is: " + str(unique_id))




file_list = sorted(os.listdir(DATA_PATH)


for idx, current_file in enumerate(files_in[:-1]):
    print(idx)
    IMAGE_1 = file_list[idx]
    IMAGE_2 = file_list[idx+1]
    IMAGE_PATH_RGB_1 = os.path.join(BASE_PATH, IMAGE_1)
    IMAGE_PATH_RGB_2 = os.path.join(BASE_PATH, IMAGE_2)


    arguments_strFirst = IMAGE_PATH_RGB_1
    arguments_strSecond = IMAGE_PATH_RGB_2

    #print('memory before inference', torch.cuda.memory_allocated())

    #imgL_o = cv2.imread(arguments_strFirst)[:,:,::-1].copy()
    #imgR_o = cv2.imread(arguments_strSecond)[:,:,::-1].copy()

    imgL_o = imread(arguments_strFirst)
    imgR_o = imread(arguments_strSecond)


    imgL_o = imgL_o[:,:,0:3]
    imgR_o = imgR_o[:,:,0:3]

    #print('mean diff{:.30f}'.format(np.mean(imgL_o_ocv - imgL_o)))
    #print('max diff{:.30f}'.format(np.max(imgL_o_ocv - imgL_o)))

    #inarr = np.expand_dims(imgL_o_ocv, axis=0)
    #print(inarr.shape)
    #ft = torch.FloatTensor(torch.from_numpy(inarr).float())


    #print(imgL_o.shape, imgL_o.dtype)
    #print(imgR_o.shape, imgR_o.dtype)




    X0 =  torch.from_numpy( np.transpose(imgL_o, (2,0,1)).astype("float32")/ 255.0).type(dtype)
    X1 =  torch.from_numpy( np.transpose(imgR_o, (2,0,1)).astype("float32")/ 255.0).type(dtype)


    y_ = torch.FloatTensor()

    assert (X0.size(1) == X1.size(1))
    assert (X0.size(2) == X1.size(2))

    #print(X0.shape, X1.shape)

    intWidth = X0.size(2)
    intHeight = X0.size(1)
    channel = X0.size(0)
    if not channel == 3:
        print('CHANNEL ERROR')

    if intWidth != ((intWidth >> 7) << 7):
        intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
        intPaddingLeft =int(( intWidth_pad - intWidth)/2)
        intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
    else:
        intWidth_pad = intWidth
        intPaddingLeft = 32
        intPaddingRight= 32

    if intHeight != ((intHeight >> 7) << 7):
        intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
        intPaddingTop = int((intHeight_pad - intHeight) / 2)
        intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
    else:
        intHeight_pad = intHeight
        intPaddingTop = 32
        intPaddingBottom = 32

    pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

    X0 = Variable(torch.unsqueeze(X0,0))
    X1 = Variable(torch.unsqueeze(X1,0))
    X0 = pader(X0)
    X1 = pader(X1)

    if use_cuda:
        X0 = X0.cuda()
        X1 = X1.cuda()
    proc_end = time.time()

    y_s,offset,filter = model(torch.stack((X0, X1),dim = 0))
            
    #print('memory after inference', torch.cuda.memory_allocated())



    y_ = y_s[save_which]


    if use_cuda:
        X0 = X0.data.cpu().numpy()
        y_ = y_.data.cpu().numpy()
        offset = [offset_i.data.cpu().numpy() for offset_i in offset]
        filter = [filter_i.data.cpu().numpy() for filter_i in filter]  if filter[0] is not None else None
        X1 = X1.data.cpu().numpy()
    else:
        X0 = X0.data.numpy()
        y_ = y_.data.numpy()
        offset = [offset_i.data.numpy() for offset_i in offset]
        filter = [filter_i.data.numpy() for filter_i in filter]
        X1 = X1.data.numpy()


    X0 = np.transpose(255.0 * X0.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
    y_ = np.transpose(255.0 * y_.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
    offset = [np.transpose(offset_i[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for offset_i in offset]

    filter = [np.transpose(
        filter_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
        (1, 2, 0)) for filter_i in filter]  if filter is not None else None
    X1 = np.transpose(255.0 * X1.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))

    print('memory after empty cache', torch.cuda.memory_allocated())

    #out_image = image_translated_with_flow_torch(imgL_o,offset[1])
    
    offset_for_saving = offset[1]#np.array(offset[1], dtype=np.float16)
    
    np.savez_compressed(os.path.join(OUTPUT_PATH, IMAGE_1+'.npz'), offset_for_saving)
    
    del X0, X1, y_s,offset,filter, y_
    gc.collect()
    torch.cuda.empty_cache()

