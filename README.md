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
JAAD leans towards an imbalance with a focus on crossing samples, while PIE has a higher proportion of non-crossing samples.

```math
JAAD      			|  NC				|  C				| \sum
-------------------------	| -------------------------	| -------------------------	| -------------------------
Train				|  400 (17.2\%)		| 1926 (82.8\%)		| 2326 (85.1\%)
Validation			|  12 (26.7\%)		| 33 (73.3\%)		| 45 (1.6\%)
Train				|  133 (17.2\%)		| 230 (82.8\%)		| 363 (13.3\%)
\sum				|  545 (19.9\%)		| 2189 (80.1\%)		| 2734
```

Additionally, the creators of the datasets have implemented a [benchmark](https://github.com/ykotseruba/PedestrianActionBenchmark) approach to enhance the comparability of models.
In this context, the observation length is set at 16 frames, equivalent to about 0.5 seconds of video, while the crossing event is projected to occur 30-60 frames (1-2 seconds) into the future (time-to-event).




## Model:
lala

## Results:
la