

# Training

**Download the dataset from here:**  [dataset]()
**Download the pretrained VGG weights here:** [pretrained VGG weights](https://www.dropbox.com/s/hp08uuzjaaz8tpm/vgg16_weights_tf_dim_ordering_tf_kernels.h5?dl=0)
**Download the Faster RCNN model weights with FC-net here for testing purposes:** [FC-net + RPN weights](https://www.dropbox.com/s/chzlhppaih64mwq/model_frcnn_fcnet.hdf5?dl=0)
**Download the Faster RCNN model weights with VGG-net here for testing purposes:** [VGG-net + RPN weights](https://www.dropbox.com/s/650f0kfdwoiuiah/model_frcnn_vgg.hdf5?dl=0)

## using pretrained VGG-net

`python train_frcnn.py --path="dataset/png_TrainIJCNN2013/gt_train.txt" --network='vgg' --hf=True --vf=True --rot=True --num_epochs=10 --epoch_length=600 --config_filename='vggnet_config.pickle' --output_weight_path='model_frcnn_vgg.hdf5'`

## using trainable FC-net

`python train_frcnn.py --path="dataset/png_TrainIJCNN2013/gt_train.txt" --network=‘fcnet’ --hf=True --vf=True --rot=True --num_epochs=10 --epoch_length=600 --config_filename=‘fcnet_config.pickle' --output_weight_path='model_frcnn_fcnet.hdf5'`




# Testing

*The results (images with bounding boxes) will be saved in results_imgs folder*

## using VGG-net

`python test_frcnn.py --path='dataset/png_TestIJCNN2013/gt_test.txt' --config_filename='vggnet_config.pickle' --network='vgg'`


## using FC-net

`python test_frcnn.py --path='dataset/png_TestIJCNN2013/gt_test.txt' --config_filename='fcnet_config.pickle' --network='fcnet'`
