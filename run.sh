#python verify_deploy.py --protofile SFD/deploy.prototxt --weightfile SFD/SFD.caffemodel --imgfile data/cat.jpg --meanB 104 --meanG 117 --meanR 123 --scale 255 --height 480 --width 960 --cuda
#python verify_train.py --protofile SFD/train.prototxt --weightfile SFD/SFD.caffemodel --cuda

#python verify_deploy.py --protofile bvlc_googlenet/deploy.prototxt --weightfile bvlc_googlenet/bvlc_googlenet.caffemodel --imgfile data/cat.jpg --meanB 104.01 --meanG 116.67 --meanR 122.68 --scale 255 --height 224 --width 224 --synset_words data/synset_words.txt --cuda
#python verify_train.py --protofile bvlc_googlenet/train_val.prototxt --weightfile bvlc_googlenet/bvlc_googlenet.caffemodel --cuda

#python verify_deploy.py --protofile resnet50/deploy.prototxt --weightfile resnet50/resnet50.caffemodel --imgfile data/cat.jpg --meanB 104.01 --meanG 116.67 --meanR 122.68 --scale 255 --height 224 --width 224 --synset_words data/synset_words.txt --cuda

#python verify_deploy.py --protofile SSD_VOC300/deploy.prototxt --weightfile SSD_VOC300/SSD_VOC300.caffemodel --imgfile data/cat.jpg --meanB 104 --meanG 117 --meanR 123 --scale 255 --height 300 --width 300 --cuda

#python verify_deploy.py --protofile SFD_fpn/SFD_fpn_deploy.prototxt --weightfile SFD_fpn/SFD_fpn.caffemodel --imgfile data/cat.jpg --meanB 104 --meanG 117 --meanR 123 --scale 255 --height 480 --width 960 --cuda


python test_caffe_loader1.py --data_protofile SFD/train_data.prototxt --net_protofile SFD/train_net.prototxt --weightfile SFD/SFD.caffemodel --cuda
python test_caffe_loader2.py --protofile SFD/train.prototxt --weightfile SFD/SFD.caffemodel --cuda
