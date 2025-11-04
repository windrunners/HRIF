
# Introduction to HRIF
HRIF is a direct haze removal image fusion method for infrared and visible images. It belongs to the subfield of infrared and visible image fusion. The algorithm can fuse the visible image with haze and infrared image, and generate the fusion image without haze.

<img width="2745" height="1281" alt="图1" src="https://github.com/user-attachments/assets/737e7e84-b886-4df6-8a45-5c04838de449" />
Fig.1 The overall algorithm

# HazeScene Dataset
HazeScene is a single mode haze dataset in which visible images contain haze and infrared images do not. The haze level of visible images includes light, medium, heavy and heavy-various levels.

HazeScene is a simulated haze dataset, rather than an actual haze scene. This dataset is a hybrid dataset, mainly derived from [RoadScene](https://github.com/hanna-xu/RoadScene), [MSRS](https://github.com/Linfeng-Tang/MSRS) and [M3FD](https://github.com/JinyuanLiu-CV/TarDAL). 

You can download the dataset from here. [HazeScene](https://github.com/windrunners/HazeScene-dataset)

# Test Model
Run "test_image.py".

# Train the model
## Train for encoder and decoder
This model employs a two-stage training approach. Firstly, the encoder and decoder are trained. Regarding how to train the encoder and decoder, you need to run the "train for encoder and decoder.py". At this point, you need to download the MSCOCO dataset and place it in the main folder.

## Train for potential mapping network
A. Before training the potential feature mapping network, it is necessary to complete the training of the encoder and decoder. Meanwhile, the optimal parameters of the encoder and decoder should be placed in the 'models' folder and named as 'TUFusion_gray.model'. 

B. For the training of the potential feature mapping network, the training dataset needs to be downloaded and placed in the "train_data" folder. 

C. This file(train for potential mapping network.py) is a function for training the potential feature mapping.

## Train for potential mapping network


# Citation
```
@article{,
    title={HRIF: haze removal image fusion for visible and infrared images},
    author={ },
    journal={},
    volume={},
    number={},
    pages={},
    year={},
    publisher={}
}
```
