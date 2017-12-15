from caffenet import *
import numpy as np

def load_synset_words(synset_file):
    lines = open(synset_file).readlines()
    synset_dict = dict()
    for i, line in enumerate(lines):
        synset_dict[i] = line.strip()
    return synset_dict

def forward_pytorch(protofile, weightfile, imgfile):
    net = CaffeNet(protofile)
    print(net)
    net.load_weights(weightfile)
    net.eval()
    #net.print_network()

    image = caffe.io.load_image(imgfile)
    if True:
        transformer = caffe.io.Transformer({'data': (1, 3, 640, 640)})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array(104, 117, 123))
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2, 1, 0))

        transformed_image = transformer.preprocess('data', image)
    image = Variable(torch.from_numpy(transformed_image).view(1,3,640,640))
    blobs = net(image)
    return blobs, net.models

# Reference from:
# http://caffe.berkeleyvision.org/gathered/examples/cpp_classification.html
# http://blog.csdn.net/zengdong_1991/article/details/52292417
def forward_caffe(protofile, weightfile, imgfile):
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(protofile, weightfile, caffe.TEST)
    image = caffe.io.load_image(imgfile)
    net.blobs['data'].reshape(1, 3, 640, 640)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    output = net.forward()
    return net.blobs, net.params

if __name__ == '__main__':
    import sys
    if False: #len(sys.argv) != 4:
        print('Usage: python caffenet.py model.prototxt model.caffemodel imgfile')
        print('')
        print('e.g. python caffenet.py ResNet-50-deploy.prototxt ResNet-50-model.caffemodel test.png')
        print('download https://github.com/daijifeng001/R-FCN/blob/master/fetch_data/fetch_model_ResNet50.m')
        exit()
    from torch.autograd import Variable
    from PIL import Image

    protofile = "SFD/deploy.prototxt" #sys.argv[1]
    weightfile = "SFD/SFD.caffemodel" #sys.argv[2]
    imgfile = "cat.jpg" #sys.argv[3]

    pytorch_blobs, pytorch_models = forward_pytorch(protofile, weightfile, imgfile)
    caffe_blobs, caffe_params = forward_caffe(protofile, weightfile, imgfile)

    # compare weights
    for layer_name in ["conv1_1"]:
        pytorch_weight = pytorch_models[layer_name].weight.data.numpy()
        pytorch_bias = pytorch_models[layer_name].bias.data.numpy()
        caffe_weight = caffe_params[layer_name][0].data
        caffe_bias = caffe_params[layer_name][1].data
        weight_diff = abs(pytorch_weight - caffe_weight).sum()
        bias_diff = abs(pytorch_bias - caffe_bias).sum()
        print('weight diff %s %f' % (layer_name, weight_diff))
        print('bias diff %s %f' % (layer_name, bias_diff))
    
    # compare outputs
    #for blob_name in ["data", "conv1/7x7_s2", "pool1/3x3_s2", "pool1/norm1", "inception_3a/output", "inception_4a/output", "inception_5a/output", "loss3/classifier"]:
    for blob_name in ["conv1_1"]:
        pytorch_data = pytorch_blobs[blob_name].data.numpy()
        caffe_data = caffe_blobs[blob_name].data
        diff = abs(pytorch_data - caffe_data).sum()
        print('output diff %s sum %f avg %f' % (blob_name, diff, diff/pytorch_data.size))
