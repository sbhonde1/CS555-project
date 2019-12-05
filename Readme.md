Framework for training a neural network model for object detection task in 2d image

### Getting started
##### Dataset
Make sure your dataset folder is in following order.

    dataset
    ├── train        
    │   ├── images          # raw images
    │   ├── labels          # bbox label files
    └── test
    │   ├── images          # raw images
    │   ├── labels          # bbox label files
    └── valid
    │   ├── images          # raw images
    │   ├── labels          # bbox label files
    └── models              # directory to save the checkpoints 
    │   ├── ...             # saved checkpoints
    └── tensorboard_logs    # directory to save tensorboard logs
        ├── ...             # saved logs 

> Please create `models` and `tensorboard_logs` directories if they don't exist.<br>
> Make sure your image file name and label file name is same 
> i.e., if image file name is `some_image.png` then corresponding label file would be `some_image.txt` 

##### Run
> Distributed
```
python -m torch.distributed.launch --nproc_per_node=<number_of_avaliable_GPUs> train.py 
--data <path_to_dataset_folder> --batch-size <N int> --log-dir <tensorboard_logs_folder> --save-as <filename>
```
> Single GPU
```
python train.py 
--data <path_to_dataset_folder> --batch-size <N int> --log-dir <tensorboard_logs_folder> --save-as <filename>
```

### References

[Object Detection PyTorch](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
[ImageNet PyTorch example](https://github.com/pytorch/examples/tree/master/imagenet)
