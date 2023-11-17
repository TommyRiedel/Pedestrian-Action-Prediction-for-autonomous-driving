# Pedestrian-Action-Prediction-for-autonomous-driving

### Learnings:
 - PyTorch (lightning)
 - (Spatio-Temporal) Graph Convolutional Neural Networks
 - Tensorhub
 
## Motivation:

Pedestrians, categorized as Vulnerable Road Users (VRUs), contribute significantly to road accident fatalities, with over 90\% of incidents attributed to human error ([WHO](https://www.who.int/health-topics/road-safety#tab=tab_1)).
Increasing vehicle automation is seen as a potential solution to reduce accidents.
Autonomous vehicles rely on sensors like cameras, LiDAR, and radar to perceive their surroundings, enabling the prediction of road users' future behaviour for early response and accident prevention.
This thesis, conducted at the Chair of Automotive Technology (TUM), focuses on developing a model to predict the crossing behaviour of pedestrians, especially crucial given their high representation among VRUs and vulnerability during road crossing.
A potential first application could be in the research project ["EDGAR"](https://www.mos.ed.tum.de/ftm/forschungsfelder/team-av-perception/edgar/), in which an attempt is being made to create a fully autonomous Wiesn shuttle.
A data-driven approach is used, as no prior information about the traffic situation will be required and the algorithm may potentially be successfully applied to complex unseen traffic scenarios.

## Data and Preprocessing:

This study utilizes the naturalistic datasets [JAAD_{beh}](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/) and [PIE](https://data.nvision2.eecs.yorku.ca/PIE_dataset/), both created by the same research group and exhibiting relative similarity. 
A notable contrast lies in the duration of recordings; JAAD comprises 5-15 second clips, while PIE consists of a continuous ten-hour recording captured on a sunny day in Toronto. 
JAAD recordings span various locations, times of day, seasons, and weather conditions across Europe and North America.
Annotation distinctions are minimal, with PIE uniquely providing precise ego-vehicle velocities. 

### Features:
#### Bounding Box location:
The bounding box is defined by two coordinates, representing the upper left and lower right corners: $b_{j} = [(x_{j}^{1}, y_{j}^{1}), (x_{j}^{2}, y_{j}^{2})]$.
This bounding box serves a dual purpose: it records the historical movement trajectory of the pedestrian and indicates their relative position within the image.
Consequently, when the bounding box is larger and more centrally located, it signifies that the pedestrian is closer to the ego vehicle

$ B_{j} = {b_{j}^{t-m+1}, b_{j}^{t-m+2}, ..., b_{j}^{t}} $

#### Ego-Vehicle velocity:
The velocity of the ego vehicle is exclusively recorded for the PIE dataset, where it is measured in kilometers per hour per frame. 
This velocity information is obtained through an On-Board Diagnostics (OBD) sensor, providing a comprehensive understanding of the ego vehicle's speed dynamics on a per-frame basis within the PIE dataset.

#### Pose Keypoint location:
The depicted human body representation relies on a simplified model consisting of 17 keypoints corresponding to major human joints (refer to the image)
However, this information is not inherent in either of the two datasets and needs to be initially estimated through a human pose estimation algorithm.
In this study, [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) is selected because of its high accuracy in keypoint estimation.
The smaller variant of the network, W-32, is chosen to optimize runtime.
The algorithm is applied to the region defined by the bounding box, which is scaled to dimensions of 256x192 pixels.
This process aims to estimate the locations of the 17 keypoints, providing a crucial foundation for subsequent analyses involving human body pose information in the datasets.

<img src="https://github.com/TommyRiedel/Pedestrian-Action-Prediction-for-autonomous-driving/assets/33426324/3010816a-2a81-40e4-95ab-2e52ac5a40f4" width="300">

$ P_{j} = {p_{j}^{t-m+1}, p_{j}^{t-m+2}, ..., p_{j}^{t}} $

Additionally, the creators of the datasets have implemented a [benchmark](https://github.com/ykotseruba/PedestrianActionBenchmark) approach to enhance the comparability of models.
In this context, the observation length is set to $m=16$ frames, equivalent to about 0.5 seconds of video, while the crossing event is projected to occur $t = [30, 60]$ frames (1-2 seconds) into the future (time-to-event).

To augment the training dataset, an overlap strategy is implemented. 
Specifically, a 60\% overlap for the PIE dataset and an 80\% overlap for JAAD are employed between sequences. 
This results in the generation of 6 sequences for PIE and 11 sequences for JAAD from a single scene. 
The utilization of this overlap approach contributes to an increased volume of training data, enhancing the model's ability to generalize and improve performance.
Furthermore, slight variations are applied to the coordinates of both the bounding box and pose keypoints.
The x-coordinates undergo variations within the range of [0, 8], while the y-coordinates vary within [0, 6].
Subsequently, these coordinates are normalized to values within the range of [0, 1], leveraging the image dimensions of height = 1920 pixels and width = 1080 pixels.
This normalization enhances numerical stability in the algorithm and facilitates faster convergence.

The distribution of data into training, validation, and testing sets is explicitly outlined in the benchmark.
The following tables present the number of data points within each respective subset. 
The COMB dataset represents a combined dataset from JAAD and PIE, and it is also subjected to analysis.
JAAD exhibits an imbalance, primarily focusing on crossing samples, whereas PIE has a higher proportion of non-crossing samples. 
Notably, PIE contains a significantly larger amount of data, approximately 2.5 times more than JAAD.

JAAD      			|  NC				|  C				| $\sum$			
-------------------------	| -------------------------	| -------------------------	| -------------------------	
Train				|  400 (17.2\%)		| 1926 (82.8\%)		| 2326 (85.1\%)		
Validation			|  12 (26.7\%)		| 33 (73.3\%)		| 45 (1.6\%)		
Train				|  133 (17.2\%)		| 230 (82.8\%)		| 363 (13.3\%)		
$\sum$			|  545 (19.9\%)		| 2189 (80.1\%)		| 2734			

PIE     			|  NC				|  C				| $\sum$
-------------------------	| -------------------------	| -------------------------	| -------------------------
Train				|  3635 (74.1\%)	| 1272 (25.9\%)		| 4907 (73.6\%)
Validation			|  358 (79.0\%)		| 95 (21.0\%)		| 453 (6.8\%)
Train				|  940 (72.0\%)		| 366 (28.0\%)		| 1306 (19.6\%)
$\sum$			|  4933 (74.0\%)	| 1733 (26.0\%)		| 6666

COMB     			|  NC				|  C				| $\sum$
-------------------------	| -------------------------	| -------------------------	| -------------------------
Train				|  4035 (55.8\%)	| 3198 (44.2\%)		| 7233 (76.9\%)
Validation			|  370 (74.3\%)		| 128 (25.7\%)		| 498 (5.3\%)
Train				|  1073 (64.3\%)	| 596 (35.7\%)		| 1669 (17.8\%)
$\sum$			|  5478 (58.3\%)	| 3922 (41.7\%)		| 9400

## Model:

![Netzwerk](https://github.com/TommyRiedel/Pedestrian-Action-Prediction-for-autonomous-driving/assets/33426324/2b01f851-96b7-47f6-85ef-ddf304fa5cbc)

## Results:

