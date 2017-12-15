import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
from prototxt import *
import caffe
import caffe.proto.caffe_pb2 as caffe_pb2

class FCView(nn.Module):
    def __init__(self):
        super(FCView, self).__init__()

    def forward(self, x):
        nB = x.data.size(0)
        x = x.view(nB,-1)
        return x
    def __repr__(self):
        return 'view(nB, -1)'

class Eltwise(nn.Module):
    def __init__(self, operation='+'):
        super(Eltwise, self).__init__()
        self.operation = operation

    def forward(self, x1, x2):
        if self.operation == '+' or self.operation == 'SUM':
            x = x1 + x2
        if self.operation == '*' or self.operation == 'MUL':
            x = x1 * x2
        if self.operation == '/' or self.operation == 'DIV':
            x = x1 / x2
        return x

class Scale(nn.Module):
    def __init__(self, channels):
        super(Scale, self).__init__()
        self.weight = Parameter(torch.Tensor(channels))
        self.bias = Parameter(torch.Tensor(channels))
        self.channels = channels

    def forward(self, x):
        nB = x.size(0)
        nC = x.size(1)
        nH = x.size(2)
        nW = x.size(3)
        x = x * self.weight.view(1, nC, 1, 1).expand(nB, nC, nH, nW) + \
            self.bias.view(1, nC, 1, 1).expand(nB, nC, nH, nW)
        return x

class Concat(nn.Module):
    def __init__(self, axis):
        super(Concat, self).__init__()
        self.axis = axis

    def forward(self, *inputs):
        return torch.cat(inputs, self.axis)

class Permute(nn.Module):
    def __init__(self, order0, order1, order2, order3):
        super(Permute, self).__init__()
        self.order0
        self.order1
        self.order2
        self.order3

    def forward(self, x):
        x = x.permute(self.order0, self.order1, self.order2, self.order3)
        return x

class L2Norm(nn.Module):
    def __init__(self,n_channels, scale=1.0):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.scale = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.weight.data *= 0.0
        self.weight.data += self.scale

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = x / norm * self.weight.view(1,-1,1,1)
        return x

class Flatten(nn.Module):
    def __init__(self, start_axis):
        super(Flatten, self).__init__()
        self.start_axis = start_axis
    def forward(self, x):
        left_size = 1
        for i in range(self.start_axis):
            left_size = x.size(i) * left_size
        return x.view(left_size, -1)

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if self.ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1), 
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0)) 
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta
    
    
    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x
 
class CaffeNet(nn.Module):
    def __init__(self, protofile):
        super(CaffeNet, self).__init__()
        self.net_info = parse_prototxt(protofile)
        self.models = self.create_network(self.net_info)
        for name,model in self.models.items():
            self.add_module(name, model)

        if self.net_info['props'].has_key('input_shape'):
            self.width = int(self.net_info['props']['input_shape']['dim'][3])
            self.height = int(self.net_info['props']['input_shape']['dim'][2])
        else:
            self.width = int(self.net_info['props']['input_dim'][3])
            self.height = int(self.net_info['props']['input_dim'][2])
        self.has_mean = False
        if self.net_info['props'].has_key('mean_file'):
            self.has_mean = True
            self.mean_file = self.net_info['props']['mean_file']

    def set_mean_file(self, mean_file):
        if mean_file != "":
            self.has_mean = True
            self.mean_file = mean_file
       
        else:
            self.has_mean = False
            self.mean_file = ""

    def forward(self, data):
        if self.has_mean:
            nB = data.data.size(0)
            nC = data.data.size(1)
            nH = data.data.size(2)
            nW = data.data.size(3)
            data = data - Variable(self.mean_img.view(1, nC, nH, nW).expand(nB, nC, nH, nW))

        blobs = OrderedDict()
        blobs['data'] = data
        
        layers = self.net_info['layers']
        layer_num = len(layers)
        i = 0
        while i < layer_num:
            layer = layers[i]
            lname = layer['name']
            ltype = layer['type']
            tname = layer['top']
            bname = layer['bottom']
            bnames = bname if type(bname) == list else [bname]
            tnames = tname if type(tname) == list else [tname]

            if ltype in ['Data', 'Accuracy', 'SoftmaxWithLoss', 'Region']:
                i = i + 1
            else:
                bdatas = [blobs[name] for name in bnames]
                tdatas = self._modules[lname](*bdatas)
                if type(tdatas) != tuple:
                    tdatas = (tdatas,)

                assert(len(tdatas) == len(tnames))
                for index, tdata in enumerate(tdatas):
                    blobs[tnames[index]] = tdata
                i = i + 1
            input_size = blobs[bnames[0]].size()
            output_size = blobs[tnames[0]].size()
            print('forward %s %s -> %s' % (lname, input_size, output_size))

        return blobs
#        if type(self.outputs) == list:
#            odatas = [blobs[name] for name in self.outputs]
#            return odatas
#        else:
#            return blobs[self.outputs]

    def print_network(self):
        print(self)
        print_prototxt(self.net_info)

    def load_weights(self, caffemodel):
        if self.has_mean:
            print('mean_file', self.mean_file)
            mean_blob = caffe_pb2.BlobProto()
            mean_blob.ParseFromString(open(self.mean_file, 'rb').read())

            if self.net_info['props'].has_key('input_shape'):
                channels = int(self.net_info['props']['input_shape']['dim'][1])
                height = int(self.net_info['props']['input_shape']['dim'][2])
                width = int(self.net_info['props']['input_shape']['dim'][3])
            else:
                channels = int(self.net_info['props']['input_dim'][1])
                height = int(self.net_info['props']['input_dim'][2])
                width = int(self.net_info['props']['input_dim'][3])

            mu = np.array(mean_blob.data)
            mu.resize(channels, height, width)
            mu = mu.mean(1).mean(1)
            mean_img = torch.from_numpy(mu).view(channels, 1, 1).expand(channels, height, width).float()
            
            self.register_buffer('mean_img', torch.zeros(channels, height, width))
            self.mean_img.copy_(mean_img)

        model = parse_caffemodel(caffemodel)
        layers = model.layer
        if len(layers) == 0:
            print('Using V1LayerParameter')
            layers = model.layers

        lmap = {}
        for l in layers:
            lmap[l.name] = l

        layers = self.net_info['layers']
        layer_num = len(layers)
        i = 0
        while i < layer_num:
            layer = layers[i]
            lname = layer['name']
            ltype = layer['type']
            if ltype == 'Convolution':
                print('load weights %s' % lname)
                convolution_param = layer['convolution_param']
                bias = True
                if convolution_param.has_key('bias_term') and convolution_param['bias_term'] == 'false':
                    bias = False
                #weight_blob = lmap[lname].blobs[0]
                #print('caffe weight shape', weight_blob.num, weight_blob.channels, weight_blob.height, weight_blob.width)
                caffe_weight = np.array(lmap[lname].blobs[0].data)
                caffe_weight = torch.from_numpy(caffe_weight).view_as(self.models[lname].weight)
                self.models[lname].weight.data.copy_(caffe_weight)
                if bias and len(lmap[lname].blobs) > 1:
                    self.models[lname].bias.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[1].data)))
                    #print("convlution %s has bias" % lname)
                i = i + 1
            elif ltype == 'BatchNorm':
                print('load weights %s' % lname)
                self.models[lname].running_mean.copy_(torch.from_numpy(np.array(lmap[lname].blobs[0].data) / lmap[lname].blobs[2].data[0]))
                self.models[lname].running_var.copy_(torch.from_numpy(np.array(lmap[lname].blobs[1].data) / lmap[lname].blobs[2].data[0]))
                i = i + 1
            elif ltype == 'Scale':
                print('load weights %s' % lname)
                self.models[lname].weight.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[0].data)))
                self.models[lname].bias.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[1].data)))
                i = i + 1
            elif ltype == 'InnerProduct':
                print('load weights %s' % lname)
                if type(self.models[lname]) == nn.Sequential:
                    self.models[lname][1].weight.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[0].data)))
                    if len(lmap[lname].blobs) > 1:
                        self.models[lname][1].bias.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[1].data)))
                else:
                    self.models[lname].weight.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[0].data)))
                    if len(lmap[lname].blobs) > 1:
                        self.models[lname].bias.data.copy_(torch.from_numpy(np.array(lmap[lname].blobs[1].data)))
                i = i + 1
            elif ltype in ['Pooling', 'Eltwise', 'ReLU', 'Region', 'Normalize', 'Permute', 'Flatten', 'Slice', 'Concat', 'Softmax', 'SoftmaxWithLoss', 'LRN', 'Dropout']:
                i = i + 1
            else:
                print('load_weights: unknown type %s' % ltype)
                i = i + 1

    def create_network(self, net_info):
        models = OrderedDict()
        blob_channels = dict()
        blob_width = dict()
        blob_height = dict()

        layers = net_info['layers']
        props = net_info['props']
        layer_num = len(layers)

        if props.has_key('input_shape'):
            blob_channels['data'] = int(props['input_shape']['dim'][1])
            blob_height['data'] = int(props['input_shape']['dim'][2])
            blob_width['data'] = int(props['input_shape']['dim'][3])
        else:
            blob_channels['data'] = int(props['input_dim'][1])
            blob_height['data'] = int(props['input_dim'][2])
            blob_width['data'] = int(props['input_dim'][3])
        i = 0
        while i < layer_num:
            layer = layers[i]
            lname = layer['name']
            ltype = layer['type']
            if ltype == 'Data':
                i = i + 1
                continue
            bname = layer['bottom']
            tname = layer['top']
            if ltype == 'Convolution':
                convolution_param = layer['convolution_param']
                channels = blob_channels[bname]
                out_filters = int(convolution_param['num_output'])
                kernel_size = int(convolution_param['kernel_size'])
                stride = int(convolution_param['stride']) if convolution_param.has_key('stride') else 1
                pad = int(convolution_param['pad']) if convolution_param.has_key('pad') else 0
                group = int(convolution_param['group']) if convolution_param.has_key('group') else 1
                bias = True
                if convolution_param.has_key('bias_term') and convolution_param['bias_term'] == 'false':
                    bias = False
                models[lname] = nn.Conv2d(channels, out_filters, kernel_size, stride,pad,group, bias=bias)
                blob_channels[tname] = out_filters
                blob_width[tname] = (blob_width[bname] + 2*pad - kernel_size)/stride + 1
                blob_height[tname] = (blob_height[bname] + 2*pad - kernel_size)/stride + 1
                i = i + 1
            elif ltype == 'BatchNorm':
                momentum = 0.9
                if layer.has_key('batch_norm_param') and layer['batch_norm_param'].has_key('moving_average_fraction'):
                    momentum = float(layer['batch_norm_param']['moving_average_fraction'])
                channels = blob_channels[bname]
                models[lname] = nn.BatchNorm2d(channels, momentum=momentum, affine=False)
                blob_channels[tname] = channels
                blob_width[tname] = blob_width[bname]
                blob_height[tname] = blob_height[bname]
                i = i + 1
            elif ltype == 'Scale':
                channels = blob_channels[bname]
                models[lname] = Scale(channels)
                blob_channels[tname] = channels
                blob_width[tname] = blob_width[bname]
                blob_height[tname] = blob_height[bname]
                i = i + 1
            elif ltype == 'ReLU':
                inplace = (bname == tname)
                if layer.has_key('relu_param') and layer['relu_param'].has_key('negative_slope'):
                    negative_slope = float(layer['relu_param']['negative_slope'])
                    models[lname] = nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
                else:
                    models[lname] = nn.ReLU(inplace=inplace)
                blob_channels[tname] = blob_channels[bname]
                blob_width[tname] = blob_width[bname]
                blob_height[tname] = blob_height[bname]
                i = i + 1
            elif ltype == 'Pooling':
                kernel_size = int(layer['pooling_param']['kernel_size'])
                stride = int(layer['pooling_param']['stride'])
                padding = 0
                if layer['pooling_param'].has_key('pad'):
                    padding = int(layer['pooling_param']['pad'])
                pool_type = layer['pooling_param']['pool']
                if pool_type == 'MAX':
                    models[lname] = nn.MaxPool2d(kernel_size, stride, padding=padding, ceil_mode=True)
                elif pool_type == 'AVE':
                    models[lname] = nn.AvgPool2d(kernel_size, stride, padding=padding, ceil_mode=True)

                if stride > 1:
                    blob_width[tname] = (blob_width[bname] + 2*padding - kernel_size + 1)/stride + 1
                    blob_height[tname] = (blob_height[bname] + 2*padding - kernel_size + 1)/stride + 1
                else:
                    blob_width[tname] = blob_width[bname] + 2*padding - kernel_size + 1
                    blob_height[tname] = blob_height[bname] + 2*padding - kernel_size + 1
                blob_channels[tname] = blob_channels[bname]
                i = i + 1
            elif ltype == 'Eltwise':
                operation = 'SUM'
                if layer.has_key('eltwise_param') and layer['eltwise_param'].has_key('operation'):
                    operation = layer['eltwise_param']['operation']
                bname0 = bname[0]
                bname1 = bname[1]
                models[lname] = Eltwise(operation)
                blob_channels[tname] = blob_channels[bname0]
                blob_width[tname] = blob_width[bname0]
                blob_height[tname] = blob_height[bname0]
                i = i + 1
            elif ltype == 'InnerProduct':
                filters = int(layer['inner_product_param']['num_output'])
                if blob_width[bname] != -1 or blob_height[bname] != -1:
                    channels = blob_channels[bname] * blob_width[bname] * blob_height[bname]
                    models[lname] = nn.Sequential(FCView(), nn.Linear(channels, filters))
                else:
                    channels = blob_channels[bname]
                    models[lname] = nn.Linear(channels, filters)
                blob_channels[tname] = filters
                blob_width[tname] = 1
                blob_height[tname] = 1
                i = i + 1
            elif ltype == 'Dropout':
                channels = blob_channels[bname]
                dropout_ratio = float(layer['dropout_param']['dropout_ratio'])
                models[lname] = nn.Dropout(dropout_ratio, inplace=True)
                blob_channels[tname] = blob_channels[bname]
                blob_width[tname] = blob_width[bname]
                blob_height[tname] = blob_height[bname]
                i = i + 1
            elif ltype == 'Normalize':
                channels = blob_channels[bname]
                scale = float(layer['norm_param']['scale_filler']['value'])
                models[lname] = L2Norm(channels, scale)
                blob_channels[tname] = blob_channels[bname]
                blob_width[tname] = blob_width[bname]
                blob_height[tname] = blob_height[bname]
                i = i + 1
            elif ltype == 'LRN':
                local_size = int(layer['lrn_param']['local_size'])
                alpha = float(layer['lrn_param']['alpha'])
                beta = float(layer['lrn_param']['beta'])
                models[lname] = LRN(local_size, alpha, beta)
                blob_channels[tname] = blob_channels[bname]
                blob_width[tname] = blob_width[bname]
                blob_height[tname] = blob_height[bname]
                i = i + 1
            elif ltype == 'Permute':
                orders = layer['permute_param']['order']
                order0 = orders[0]
                order1 = orders[1]
                order2 = orders[2]
                order3 = orders[3]
                models[lname] = Permute(order0, order1, order2, order3)
                shape = torch.IntTenor([1, blob_channels[bname], blob_width[bname], blob_height[bname]])
                shape = shape.permute(order0, order1, order2, order3)
                blob_channels[tname] = shape[order1]
                blob_height[tname] = shape[order2]
                blob_width[tname] = shape[order3]
                i = i + 1
            elif ltype == 'Flatten':
                axis = int(layer['flatten_param']['axis'])
                models[lname] = Flatten(axis)
                blob_channels[tname] = blob_channels[bname] * blob_width[bname] * blob_height[bname]
                blob_width[tname] = 1
                blob_height[tname] = 1
                i = i + 1
            elif ltype == 'Slice':
                axis = int(layer['slice_param']['axis'])
                slice_points = layer['slice_param']['slice_point']
                models[lname] = Slice(axis, slice_points)
                for tn in tname:
                    blob_channels[tn] = 1
                    blob_width[tn] = blob_width[bname]
                    blob_height[tn] = blob_height[bname]
            elif ltype == 'Concat':
                axis = 1
                if layer.has_key('concat_param') and layer['concat_param'].has_key('axis'):
                    axis = int(layer['concat_param']['axis'])
                models[lname] = Concat(axis)  
                if axis == 1:
                    blob_channels[tname] = 0
                    for bn in bname:
                        blob_channels[tname] += blob_channels[bn]
                        blob_width[tname] = blob_width[bn]
                        blob_height[tname] = blob_height[bn]
                i = i + 1
            elif ltype == 'Softmax':
                models[lname] = nn.Softmax()
                blob_channels[tname] = blob_channels[bname]
                blob_width[tname] = 1
                blob_height[tname] = 1
                i = i + 1
            elif ltype == 'SoftmaxWithLoss':
                loss = nn.CrossEntropyLoss()
                blob_width[tname] = 1
                blob_height[tname] = 1
                i = i + 1
            elif ltype == 'Region':
                anchors = layer['region_param']['anchors'].strip('"').split(',')
                self.anchors = [float(j) for j in anchors]
                self.num_anchors = int(layer['region_param']['num'])
                self.anchor_step = len(self.anchors)/self.num_anchors
                self.num_classes = int(layer['region_param']['classes'])
                i = i + 1
            else:
                print('create_network: unknown type #%s#' % ltype)
                i = i + 1
            input_width = blob_width[bname] if type(bname) != list else blob_width[bname[0]]
            input_height = blob_height[bname] if type(bname) != list else blob_height[bname[0]]
            output_width = blob_width[tname] if type(tname) != list else blob_width[tname[0]]
            output_height = blob_height[tname] if type(tname) != list else blob_height[tname[0]]
            print('create %s (%d x %d) -> (%d x %d)' % (lname, input_width, input_height, output_width, output_height))

        return models


