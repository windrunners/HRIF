
# 1 Introduction to HRIF
HRIF is a direct haze removal image fusion method for infrared and visible images. It belongs to the subfield of infrared and visible image fusion. The algorithm can fuse the visible image with haze and infrared image, and generate the fusion image without haze.

<img width="2745" height="1281" alt="图1" src="https://github.com/user-attachments/assets/737e7e84-b886-4df6-8a45-5c04838de449" />
Fig.1 The overall algorithm

# 2 HazeScene Dataset
HazeScene is a single mode haze dataset in which visible images contain haze and infrared images do not. The haze level of visible images includes light, medium, heavy and heavy-various levels.

HazeScene is a simulated haze dataset, rather than an actual haze scene. This dataset is a hybrid dataset, mainly derived from [RoadScene](https://github.com/hanna-xu/RoadScene), [MSRS](https://github.com/Linfeng-Tang/MSRS) and [M3FD](https://github.com/JinyuanLiu-CV/TarDAL). 

You can download the dataset from here. [HazeScene](https://github.com/windrunners/HazeScene-dataset)

# 3 Test Model
Run "test_image.py".

# 4 Train model
## 4.1 Train for encoder and decoder
This model employs a two-stage training approach. Firstly, the encoder and decoder are trained. Regarding how to train the encoder and decoder, you need to run the "train for encoder and decoder.py". At this point, you need to download the MSCOCO dataset and place it in the main folder.

## 4.2 Train for potential mapping network
A. Before training the potential feature mapping network, it is necessary to complete the training of the encoder and decoder. Meanwhile, the optimal parameters of the encoder and decoder should be placed in the 'models' folder and named as 'TUFusion_gray.model'. 

B. For the training of the potential feature mapping network, the training dataset needs to be downloaded and placed in the "train_data" folder. 

C. This file(train for potential mapping network.py) is a function for training the potential feature mapping.

# 5 Average smoke concentration (ASC) evaluation metric
The calculation method of ASC is as follows:
```
import cv2
import numpy as np
import os
def compute_dark_channel(img, patch_size=15):
    """Calculate the dark channel"""
    min_channel = np.min(img, axis=2)  # Take the minimum value of the three channels (Red, Green, Blue)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)  # Minimum value filtering
    return dark_channel
def calculate_smoke_concentration(img):
    """Calculate the smoke concentration of a single image (normalize the dark channel to the range of 0-1)"""
    dark = compute_dark_channel(img)
    concentration = np.mean(dark) / 255.0  # Standardized to [0, 1]
    return concentration
def batch_calculate_smoke(image_folder, output_file="smoke_concentration_results.txt"):
    """Calculate the smoke concentration (0-1) of all images in the batch folder"""
    concentrations = []
    results = []
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            concentration = calculate_smoke_concentration(img)
            concentrations.append(concentration)
            results.append(f"{filename}: {concentration:.4f}")  # Keep 4 decimal places
    # Calculate the average value
    avg_concentration = np.mean(concentrations) if concentrations else 0
    results.append(f"\nAverage smoke concentration: {avg_concentration:.4f}")
    # Save the results to a file
    with open(output_file, "w") as f:
        f.write("\n".join(results))
    print(f"Results saved to {output_file}")
    print(f"Average smoke concentration: {avg_concentration:.4f}")
    return concentrations, avg_concentration
# Calculation
image_folder = "your_image_folder"  # Replace with the path of your image folder
concentrations, avg = batch_calculate_smoke(image_folder)
```

# Citation
```
@article{zhao2025hrif,
  title={HRIF: haze removal image fusion for visible and infrared images},
  author={Zhao, Yangyang and Li, Wenjun and Yu, Zhiyong},
  journal={Measurement Science and Technology},
  volume={36},
  number={11},
  pages={115403},
  year={2025}
}
```
