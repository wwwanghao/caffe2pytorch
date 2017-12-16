## caffe2pytorch
This tool aims to load caffe prototxt and weights directly in pytorch without explicitly converting model from caffe to pytorch.

### Usage
```
from caffenet import *

def load_image(imgfile):
    image = caffe.io.load_image(imgfile)
    transformer = caffe.io.Transformer({'data': (1, 3, args.height, args.width)})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([args.meanB, args.meanG, args.meanR]))
    transformer.set_raw_scale('data', args.scale)
    transformer.set_channel_swap('data', (2, 1, 0))

    image = transformer.preprocess('data', image)
    image = image.reshape(1, 3, args.height, args.width)
    return image

def forward_pytorch(protofile, weightfile, image):
    net = CaffeNet(protofile)
    print(net)
    net.load_weights(weightfile)
    net.eval()
    image = torch.from_numpy(image)
    image = Variable(image)
    blobs = net(image)
    return blobs, net.models

imgfile = 'cat.jpg'
protofile = 'resnet50/deploy.prototxt'
weightfile = 'resnet50/resnet50.caffemodel'
image = load_image(imgfile)
pytorch_blobs, pytorch_models = forward_pytorch(protofile, weightfile, image)

```

### Supported Layers
Each layer in caffe will have a corresponding layer in pytorch. 
- [x] Convolution
- [x] InnerProduct
- [x] BatchNorm
- [x] Scale
- [x] ReLU
- [x] Pooling
- [x] Reshape
- [x] Softmax
- [x] SoftmaxWithLoss. 
- [x] Dropout
- [x] Eltwise
- [x] Normalize
- [x] Permute
- [x] Flatten
- [x] Slice
- [x] Concat
- [x] PriorBox
- [ ] LRN

### Verify between caffe and pytorch
python verify.py --protofile resnet50/deploy.prototxt --weightfile resnet50/resnet50.caffemodel --imgfile cat.jpg --meanB 104.01 --meanG 116.67 --meanR 122.68 --scale 255 --height 224 --width 224 --synset_words synset_words.txt

Note: synset_words.txt is the description of class names, each line represention a class name.

Outputs:
```
------------ Parameter Difference ------------
conv1                          weight_diff: 0.000000 bias_diff: 0.000000
res2a_branch1                  weight_diff: 0.000000
res2a_branch2a                 weight_diff: 0.000000
res2a_branch2b                 weight_diff: 0.000000
res2a_branch2c                 weight_diff: 0.000000
res2b_branch2a                 weight_diff: 0.000000
res2b_branch2b                 weight_diff: 0.000000
res2b_branch2c                 weight_diff: 0.000000
res2c_branch2a                 weight_diff: 0.000000
res2c_branch2b                 weight_diff: 0.000000
res2c_branch2c                 weight_diff: 0.000000
res3a_branch1                  weight_diff: 0.000000
res3a_branch2a                 weight_diff: 0.000000
res3a_branch2b                 weight_diff: 0.000000
res3a_branch2c                 weight_diff: 0.000000
res3b_branch2a                 weight_diff: 0.000000
res3b_branch2b                 weight_diff: 0.000000
res3b_branch2c                 weight_diff: 0.000000
res3c_branch2a                 weight_diff: 0.000000
res3c_branch2b                 weight_diff: 0.000000
res3c_branch2c                 weight_diff: 0.000000
res3d_branch2a                 weight_diff: 0.000000
res3d_branch2b                 weight_diff: 0.000000
res3d_branch2c                 weight_diff: 0.000000
res4a_branch1                  weight_diff: 0.000000
res4a_branch2a                 weight_diff: 0.000000
res4a_branch2b                 weight_diff: 0.000000
res4a_branch2c                 weight_diff: 0.000000
res4b_branch2a                 weight_diff: 0.000000
res4b_branch2b                 weight_diff: 0.000000
res4b_branch2c                 weight_diff: 0.000000
res4c_branch2a                 weight_diff: 0.000000
res4c_branch2b                 weight_diff: 0.000000
res4c_branch2c                 weight_diff: 0.000000
res4d_branch2a                 weight_diff: 0.000000
res4d_branch2b                 weight_diff: 0.000000
res4d_branch2c                 weight_diff: 0.000000
res4e_branch2a                 weight_diff: 0.000000
res4e_branch2b                 weight_diff: 0.000000
res4e_branch2c                 weight_diff: 0.000000
res4f_branch2a                 weight_diff: 0.000000
res4f_branch2b                 weight_diff: 0.000000
res4f_branch2c                 weight_diff: 0.000000
res5a_branch1                  weight_diff: 0.000000
res5a_branch2a                 weight_diff: 0.000000
res5a_branch2b                 weight_diff: 0.000000
res5a_branch2c                 weight_diff: 0.000000
res5b_branch2a                 weight_diff: 0.000000
res5b_branch2b                 weight_diff: 0.000000
res5b_branch2c                 weight_diff: 0.000000
res5c_branch2a                 weight_diff: 0.000000
res5c_branch2b                 weight_diff: 0.000000
res5c_branch2c                 weight_diff: 0.000000
------------ Output Difference ------------
data                           output_diff: 0.000000
conv1                          output_diff: 0.000000
pool1                          output_diff: 0.000000
res2a_branch1                  output_diff: 0.000000
res2a_branch2a                 output_diff: 0.000000
res2a_branch2b                 output_diff: 0.000000
res2a_branch2c                 output_diff: 0.000000
res2a                          output_diff: 0.000000
res2b_branch2a                 output_diff: 0.000000
res2b_branch2b                 output_diff: 0.000000
res2b_branch2c                 output_diff: 0.000000
res2b                          output_diff: 0.000000
res2c_branch2a                 output_diff: 0.000000
res2c_branch2b                 output_diff: 0.000001
res2c_branch2c                 output_diff: 0.000001
res2c                          output_diff: 0.000001
res3a_branch1                  output_diff: 0.000001
res3a_branch2a                 output_diff: 0.000000
res3a_branch2b                 output_diff: 0.000000
res3a_branch2c                 output_diff: 0.000001
res3a                          output_diff: 0.000000
res3b_branch2a                 output_diff: 0.000000
res3b_branch2b                 output_diff: 0.000000
res3b_branch2c                 output_diff: 0.000000
res3b                          output_diff: 0.000000
res3c_branch2a                 output_diff: 0.000000
res3c_branch2b                 output_diff: 0.000000
res3c_branch2c                 output_diff: 0.000000
res3c                          output_diff: 0.000000
res3d_branch2a                 output_diff: 0.000000
res3d_branch2b                 output_diff: 0.000000
res3d_branch2c                 output_diff: 0.000001
res3d                          output_diff: 0.000001
res4a_branch1                  output_diff: 0.000001
res4a_branch2a                 output_diff: 0.000000
res4a_branch2b                 output_diff: 0.000000
res4a_branch2c                 output_diff: 0.000001
res4a                          output_diff: 0.000000
res4b_branch2a                 output_diff: 0.000000
res4b_branch2b                 output_diff: 0.000000
res4b_branch2c                 output_diff: 0.000000
res4b                          output_diff: 0.000000
res4c_branch2a                 output_diff: 0.000000
res4c_branch2b                 output_diff: 0.000000
res4c_branch2c                 output_diff: 0.000000
res4c                          output_diff: 0.000000
res4d_branch2a                 output_diff: 0.000000
res4d_branch2b                 output_diff: 0.000000
res4d_branch2c                 output_diff: 0.000000
res4d                          output_diff: 0.000000
res4e_branch2a                 output_diff: 0.000000
res4e_branch2b                 output_diff: 0.000000
res4e_branch2c                 output_diff: 0.000000
res4e                          output_diff: 0.000000
res4f_branch2a                 output_diff: 0.000000
res4f_branch2b                 output_diff: 0.000000
res4f_branch2c                 output_diff: 0.000000
res4f                          output_diff: 0.000000
res5a_branch1                  output_diff: 0.000002
res5a_branch2a                 output_diff: 0.000000
res5a_branch2b                 output_diff: 0.000000
res5a_branch2c                 output_diff: 0.000001
res5a                          output_diff: 0.000000
res5b_branch2a                 output_diff: 0.000000
res5b_branch2b                 output_diff: 0.000000
res5b_branch2c                 output_diff: 0.000001
res5b                          output_diff: 0.000000
res5c_branch2a                 output_diff: 0.000000
res5c_branch2b                 output_diff: 0.000000
res5c_branch2c                 output_diff: 0.000001
res5c                          output_diff: 0.000000
pool5                          output_diff: 0.000000
fc1000                         output_diff: 0.000001
prob                           output_diff: 0.000000
------------ Classification ------------
pytorch classification top1: 0.193018 n02113023 Pembroke, Pembroke Welsh corgi
caffe   classification top1: 0.193018 n02113023 Pembroke, Pembroke Welsh corgi
```
