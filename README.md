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
This is a binary classification problem $j = {0, 1}$, where crossing (C) is usually regarded as 1 and non-crossing (NC) as 0.
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
This is a top-down multi-person pose estimation algorithm that follows a sequential process. 
Initially, individuals are identified through a person detector, and subsequently, their poses are estimated individually. 
The algorithm maintains a high-resolution representation throughout the entire process, incorporating stepwise high-to-low resolution subnetworks. 
These multi-resolution subnetworks operate in parallel, facilitating repeated multiscale fusions.
The parallel connections enable high-to-low-resolution representations to receive information from other concurrent representations.
The predicted keypoint heatmap is potentially more accurate and spatially precise. 
The smaller variant of the network, W-32, is chosen to optimize runtime.
The algorithm is applied to the region defined by the bounding box, which is scaled to dimensions of 256x192 pixels.
This process aims to estimate the locations of the 17 keypoints, providing a crucial foundation for subsequent analyses involving human body pose information in the datasets.

<p align="center">
<img src="https://github.com/TommyRiedel/Pedestrian-Action-Prediction-for-autonomous-driving/assets/33426324/3010816a-2a81-40e4-95ab-2e52ac5a40f4" width="450">
</p>

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
The main building block of the network is a Spatio-Temporal Graph Convolutional Network (ST-GCN).
This block is applied to both the bounding box and the pose keypoint location inputs.
It comprises a spatial graph convolution, extracting information between spatially neighbouring keypoints within a frame, and a temporal convolution, considering the temporal evolution of a keypoint's position in consecutive frames.
The keypoint connections within the graph can be trained, initially starting from natural connections as illustrated in the image above. 
However, these connections can be adapted based on the specific data and the problem at hand, potentially leading to improved connections.
The ego vehicle velocity is processed using temporal convolution layers. 
The outputs from the three branches of the network are then combined and weighted using an attention block. 
This block assigns higher weights to features that contribute more significantly to the classification task. 
Finally, the classification into the two classes is performed using two consecutive fully connected layers.
Through experimentation, it was found that employing two ST-GCN blocks yielded the best results.

<p align="center">
<img src="https://github.com/TommyRiedel/Pedestrian-Action-Prediction-for-autonomous-driving/assets/33426324/2b01f851-96b7-47f6-85ef-ddf304fa5cbc">
</p>

## Results:
#### PIE ablation study:
Initially, an ablation study was conducted on the PIE dataset to explore the effectiveness of various features or combinations thereof. 
For approaches "P1-3", three different implementations of the ST-GCN block were examined.
"V" denotes ego-vehicle velocity and "B" indicates bounding box location.
The results in the corresponding folder indicate that the ego vehicle velocity has the highest significance on the model's performance.
However, using ego-vehicle velocity as input raises concerns, as it is also an indirect output of the algorithm and reflects the driver's response (data recorded with human drivers) to pedestrians, a factor absent in fully autonomous vehicles or when the driver fails to recognize pedestrians in critical situations.
Consequently, contrary to this study and much of the literature, this feature should be excluded.
The use of 17 keypoints, as opposed to the two points of the bounding box, hardly yields superior results. 
Therefore, the consideration of pose keypoints with the ST-GCN block seems to offer little added value and warrants further investigation. 
Additionally, the fusion of features requires revision, as it leads to poorer results despite the additional information.

#### JAAD and COMB Dataset and Cross-evaluation:
As can be seen from the corresponding folder (results), the results with the PIE data set are significantly better than with JAAD or the combined dataset.
This enhancement can be attributed, in part, to the larger volume of data within this dataset. 
Another contributing factor may be the more consistent environmental conditions in PIE. 
However, it's crucial for Level 5 vehicles to handle diverse conditions, necessitating data that includes scenarios like snowy conditions.
The combined data set does not show particularly good results either, as can be seen from the cross-evaluation.
For the JAAD data the model trained on the combined dataset tends to favour crossing, while non-crossing is overestimated for PIE data.
This aligns with the overrepresented class in each dataset, indicating a tendency to underestimate the underrepresented class relatively strongly.
PIE shows relatively good generalizability to the JAAD dataset.