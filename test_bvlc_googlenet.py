from caffenet import *
import numpy as np

def load_synset_words(synset_file):
    lines = open(synset_file).readlines()
    synset_dict = dict()
    for i, line in enumerate(lines):
        synset_dict[i] = line.strip()
    return synset_dict

def forward_pytorch(protofile, weightfile, meanfile, imgfile):
    net = CaffeNet(protofile)
    print(net)
    net.load_weights(weightfile)
    net.eval()
    #net.print_network()

    image = caffe.io.load_image(imgfile)
    if True:
        mean_blob = caffe_pb2.BlobProto()
        mean_blob.ParseFromString(open(meanfile, 'rb').read())
        mu = np.array(mean_blob.data)
        mu = mu.reshape(mean_blob.channels, mean_blob.height, mean_blob.width)
        mu = mu.mean(1).mean(1)
        transformer = caffe.io.Transformer({'data': (1, 3, 224, 224)})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', mu)
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2, 1, 0))

        transformed_image = transformer.preprocess('data', image)
    image = Variable(torch.from_numpy(transformed_image).view(1,3,224,224))
    output = net(image)
    return output

# Reference from:
# http://caffe.berkeleyvision.org/gathered/examples/cpp_classification.html
# http://blog.csdn.net/zengdong_1991/article/details/52292417
def forward_caffe(protofile, weightfile, meanfile, imgfile):
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(protofile, weightfile, caffe.TEST)
    image = caffe.io.load_image(imgfile)
    net.blobs['data'].reshape(1, 3, 224, 224)

    mean_blob = caffe_pb2.BlobProto()
    mean_blob.ParseFromString(open(meanfile, 'rb').read())
    mu = np.array(mean_blob.data)
    print(mu.shape, mean_blob.channels, mean_blob.height, mean_blob.width)
    mu = mu.reshape(mean_blob.channels, mean_blob.height, mean_blob.width)
    #mu = np.load(meanfile)
    mu = mu.mean(1).mean(1)
    print ('mean-subtracted values:', zip('BGR', mu))

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', mu)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    output = net.forward()
    return net.blobs

if __name__ == '__main__':
    import sys
    if False: #len(sys.argv) != 4:
        print('Usage: python caffenet.py model.prototxt model.caffemodel imgfile')
        print('')
        print('e.g. python caffenet.py ResNet-50-deploy.prototxt ResNet-50-model.caffemodel test.png')
        exit()
    from torch.autograd import Variable
    from PIL import Image

    protofile = "bvlc_googlenet/deploy.prototxt" #sys.argv[1]
    weightfile = "bvlc_googlenet/bvlc_googlenet.caffemodel" #sys.argv[2]
    imgfile = "cat.jpg" #sys.argv[3]
    meanfile = "bvlc_googlenet/imagenet_mean.binaryproto" #"ResNet_mean.binaryproto"

    pytorch_blobs = forward_pytorch(protofile, weightfile, meanfile, imgfile)
    caffe_blobs = forward_caffe(protofile, weightfile, meanfile, imgfile)


    for blob_name in ["data", "conv1/7x7_s2", "inception_3a/output", "inception_4a/output", "inception_5a/output", "loss3/classifier"]:
        pytorch_data = pytorch_blobs[blob_name].data.numpy()
        caffe_data = caffe_blobs[blob_name].data
        diff = abs(pytorch_data - caffe_data).sum()
        print('diff %s %f' % (blob_name, diff))

    if True:
        synset_dict = load_synset_words('synset_words.txt')
        pytorch_prob = pytorch_blobs['prob'].data.view(-1).numpy()
        caffe_prob = caffe_blobs['prob'].data[0]
        print(pytorch_prob.max(), synset_dict[pytorch_prob.argmax()])
        print(caffe_prob.max(), synset_dict[caffe_prob.argmax()])
