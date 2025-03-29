# Introduction to HRIF
HRIF is a direct haze removal image fusion method for infrared and visible images. It belongs to the subfield of infrared and visible image fusion. The algorithm can fuse the visible image with haze and infrared image, and generate the fusion image without haze.

# HazeScene Dataset
HazeScene is a single mode haze dataset in which visible images contain haze and infrared images do not. The haze level of visible images includes light, medium and heavy levels.

HazeScene is a simulated haze dataset, rather than an actual haze scene. This dataset is a hybrid dataset, mainly derived from [RoadScene](https://github.com/hanna-xu/RoadScene) and [MSRS](https://github.com/Linfeng-Tang/MSRS). For the generation of the three haze levels, the [AdverseWeatherSimulation](https://github.com/RicardooYoung/AdverseWeatherSimulation) program was used.

For any of the light, medium and heavy levels, there are 582 pairs of images within HazeScene.
