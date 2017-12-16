## caffe2pytorch
This tool aims to load caffe prototxt and weights directly in pytorch without explicitly converting model from caffe to pytorch. Each layer in caffe will have a corresponding layer in pytorch. Currently supported layers including Convolution, BatchNorm, Scale, ReLU, Pooling, Eltwise, InnerProduct, Dropout, Normalize, LRN, Permute, Flatten, Slice, Concat, PriorBox, Reshape, Softmax, SoftmaxWithLoss. 

### Verify between caffe and pytorch
python verify.py --protofile resnet50/deploy.prototxt --weightfile resnet50/resnet50.caffemodel --imgfile cat.jpg --meanB 104.01 --meanG 116.67 --meanR 122.68 --scale 255 --height 224 --width 224 --synset_words synset_words.txt

Note: synset_words.txt is the description of class names, each line represention a class name.

