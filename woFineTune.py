import numpy as np
import matplotlib.pyplot as plt
import caffe
import os
import sys
import lmdb
import warnings
from numpy import linalg as LA
from sklearn import metrics
warnings.filterwarnings("ignore")
from preprocess import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

''' set caffe environment '''
caffe_root = 'F:/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print('CaffeNet found.')
else:
    print('Downloading pre-trained CaffeNet model...')

''' without finetune alexnet model path'''
model_def = caffe_root + 'models/bvlc_alexnet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'

''' finetune alexnet model path'''
# model_def = caffe_root + 'models/twoalexnet/deploy.prototxt'
# model_weights = caffe_root + 'models/twoalexnet/alex_finetune.caffemodel'

# ''' without finetune alexnet model path'''
# model_def = caffe_root + 'models/vgg16/deploy.prototxt'
# model_weights = caffe_root + 'models/vgg16/vgg16.caffemodel'

# ''' finetune alexnet model path'''
# model_def = caffe_root + 'models/vgg16/deploy.prototxt'
# model_weights = caffe_root + 'models/twovgg16/vgg16_finetune.caffemodel'

#define caffe nets
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
# mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
# mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
mu = BGR_mean('LFW_mean_image.npy')
print('mean-subtracted values:', list(zip('BGR', mu)))

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(10,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()
# caffe.set_mode_cpu()

def train_image(net,feature_path,feature_layer):
    prefix = "F:/Notes/Rutgers/2017_spring/DSPadv/project2/lfw-funneled/lfw_funneled/"
    pair_path = "F:/Notes/Rutgers/2017_spring/DSPadv/project2/lfw-funneled/lfw_funneled/pairs_01.txt"
    files = get_pairs_list(prefix,pair_path)
    i = 0
    j = 0
    cnt = 0
    features_np = np.zeros((1200,1000))

    for f in files:
        if(os.path.isfile(f)):
            if(i == 10):
                i=0
                j=j+1
                # now we get 50 image data input
                # we will forward compute it
                # and extract features
                out = net.forward()
                featureData = net.blobs[feature_layer].data

                for k in range(0,10):
                    feature=featureData[k].reshape(1,-1)
                    features_np[cnt,:] = feature
                    cnt+=1
                print('@_@ have extracted ',j*10,' images ')
        # save file name
        # read file to data
        # index + 1
        net.blobs['data'].data[i,:,:,:] = transformer.preprocess('data', caffe.io.load_image(f))
        if i == 9:
            print(net.blobs['data'].data.shape)
        i=i+1
    np.save(feature_path,features_np)

def read_features(featurenp_dir):
    #read data
    features = np.load(featurenp_dir)
    r,c = features.shape

    #compute l2 norm
    even_row = features[::2]
    odd_row =features[1::2]
    difference_row = even_row - odd_row
    scores = LA.norm(difference_row, axis=1)
    label = np.zeros(int(r/2))
    label[::2] = np.ones(int(r/4))

    #get roc data
    fpr, tpr, thresholds = metrics.roc_curve(label, scores, pos_label=0)
    roc_auc = metrics.auc(fpr, tpr)

    #plot roc
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()

    return fpr, tpr, roc_auc

#train_image(net,'features_np_1.npy','fc8')
#train_image(net,'vgg_features_np_2.npy','fc7')
#train_image(net,'vgg_features_np_2.npy','fc8')

# read_features('vgg_features_np_2.npy')



fpr_list = []
tpr_list = []
roc_auc_list = []

fpr, tpr, roc_auc = read_features('vgg_features_np_finetune_1.npy')
fpr_list.append(fpr)
tpr_list.append(tpr)
roc_auc_list.append(roc_auc)

fpr, tpr, roc_auc = read_features('vgg_features_np_1.npy')
fpr_list.append(fpr)
tpr_list.append(tpr)
roc_auc_list.append(roc_auc)

fpr, tpr, roc_auc = read_features('features_np_finetune_1.npy')
fpr_list.append(fpr)
tpr_list.append(tpr)
roc_auc_list.append(roc_auc)

fpr, tpr, roc_auc = read_features('features_np_1.npy')
fpr_list.append(fpr)
tpr_list.append(tpr)
roc_auc_list.append(roc_auc)

# plot roc
#colors = ['aqua', 'darkorange', ]
netname = ['vgg_finetune','vgg','alexnet_finetune','alexnet']
for i in range(4):
    plt.plot(fpr_list[i], tpr_list[i],label='ROC curve of {0} (area = {1:0.2f})'.format(netname[i], roc_auc_list[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Face Verification with Different Neural Network')
plt.legend(loc="lower right")
plt.show()
