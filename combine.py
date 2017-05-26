#coding:utf-8
import numpy as np
import sys, os
import caffe

# Edit the paths as needed:
caffe_root = 'F:/caffe/'

import caffe

# Path to your combined net prototxt files:
caffe.set_mode_cpu()

''' alex net '''
# model_def = caffe_root + 'models/bvlc_alexnet/deploy.prototxt'
# model_weights = caffe_root + 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'
#
# combined_proto = caffe_root + 'models/twoalexnet/twoalex_train_val.prototxt'


''' vgg net '''
model_def = caffe_root + 'models/vgg16/deploy.prototxt'
model_weights = caffe_root + 'models/vgg16/vgg16.caffemodel'

combined_proto = caffe_root + 'models/twovgg16/twovgg_train_val.prototxt'


# The pre-trained Caffemodel files:
net_file = model_weights
# Their respective prototxt files:
net_proto = model_def

# Chdir if your prototxt files specify your training and testing files
# in relative paths:
net = caffe.Net(net_proto, net_file, caffe.TRAIN)
comb_net = caffe.Net(combined_proto, caffe.TRAIN)

# The layers you want to combine into the new caffe net:
#alex net
# layer_names = ['conv1','conv2', 'conv3','conv4','conv5','fc6','fc7','fc8']

#vgg
layer_names = ['conv1_1', 'conv1_2',
		  'conv2_1', 'conv2_2',
		  'conv3_1', 'conv3_2', 'conv3_3',
		  'conv4_1', 'conv4_2', 'conv4_3',
		  'conv5_1', 'conv5_2', 'conv5_3',
		  'fc6', 'fc7', 'fc8']

# For each of the pretrained net sides, copy the params to
# the corresponding layer of the combined net:
for layer in layer_names:
    W = net.params[layer][0].data[...] # Grab the pretrained weights
    b = net.params[layer][1].data[...] # Grab the pretrained bias
    # comb_net.params['{}_{}'.format(side, layer)][0].data[...] = W # Insert into new combined net
    # comb_net.params['{}_{}'.format(side, layer)][1].data[...] = b
    comb_net.params['{}_p'.format(layer)][0].data[...] = W
    comb_net.params['{}_p'.format(layer)][1].data[...] = b
    comb_net.params[layer][0].data[...] = W
    comb_net.params[layer][1].data[...] = b

#chech data
#alex
# print(comb_net.params['conv1'][0].data)
# print(comb_net.params['conv1'][1].data - lnet.params['conv1'][1].data)
# print(comb_net.params['conv1_p'][1].data - lnet.params['conv1'][1].data)

#vgg
print(comb_net.params['conv1_1'][0].data)
print('\n')
print(comb_net.params['fc6_p'][1].data - net.params['fc6'][1].data)
print(comb_net.params['fc6'][1].data - net.params['fc6'][1].data)

comb_net.save('twovgg16.caffemodel')