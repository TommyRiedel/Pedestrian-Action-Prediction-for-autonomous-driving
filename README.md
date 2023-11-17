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

Additionally, the creators of the datasets have implemented a [benchmark](https://github.com/ykotseruba/PedestrianActionBenchmark) approach to enhance the comparability of models.
In this context, the observation length is set to $m=16$ frames, equivalent to about 0.5 seconds of video, while the crossing event is projected to occur $t = [30, 60]$ frames (1-2 seconds) into the future (time-to-event).

Um mehr Trainingsdaten zur Verfügung zu haben wird ein Overlap von 60\% für PIE bzw. 80\% für JAAD zwischen den Sequenzen verwendet.
Daraus folgt, dass aus einer Szene 6 bzw. 11 Sequenzen und damit Daten erzeugt werden.

Um die Robustheit des Algorithmus zu erhöhen werden die Koordinaten von Bounding Box und pose keypoints leicht variiert (Augmentation).
Die x-Koordinaten werden um [0, 8] und die x-Koordinaten um [0, 6] variiert.
Außerdem werden die Koordinaten im Anschluss mit Hilfe der height = 1920 pixel und width = 1080 pixel auf Werte [0, 1] normalisiert.
This improves the numerical stability of the algorithm and leads to faster convergence.

Die Aufteilung der Daten in Trainings, Validierung und Test wird in der benchmark ebenfalls vorgegeben.
Die nachfolgenden Tabellen zeigen die Anzahl der Daten in den jeweiligen subsets.
COMB stellt dabei einen aus JAAD und PIE kombinierten Datensatz dar, der ebenfalls untersucht wird.
Es ist zu erkennen, dass PIE deutlich mehr Daten beinhaltet (2,5 mal so viele).

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

## Features:
### Bounding Box location:
Die Bounding Box eines Fußgängers wird mit 2 Koordinaten, der linken oberen und der rechten unteren angegeben: $b_{j} = [(x_{j}^{1}, y_{j}^{1}), (x_{j}^{2}, y_{j}^{2})]$.
In ihr sind zum einen die vergangene Bewegungstrajektorie des Fußgängers sowie die relative Position des Fußgängers im Bild gespeichert.
Demnach befindet sich der Fußgänger näher am ego Fahrzeug wenn die Bounding Box größer und zentraler im Bild ist.

$ B_{j} = {b_{j}^{t-m+1}, b_{j}^{t-m+2}, ..., b_{j}^{t}} $

#### Ego-Vehicle velocity:
Die ego vehicle velocity ist nur für den PIE Datensatz gespeichert, dort wurde sie mittels OBD sensor in km/h per frame gemessen.

### Pose Keypoint location:
Dabei handelt es sich um eine vereinfachte Darstellung des menschlichen Körpers anhand 17 keypoints, die den major human joints entsprechen (siehe Bild).
Diese Information ist in keinem der beiden Datensätze gespeichert und muss daher zunächst mit Hilfe eines human pose estimation algorithmus geschätzt werden.
Hierfür wird in dieser Arbeit [HRNet]() verwendet, da dieses mit die genausten Schätzungen für die keypoints liefert.
Aufgrund der geringeren Laufzeit wird die kleine Variante des Netzwerks (W-32) verwendet.
Der Algorithmus wird auf den Ausschnitt der Bounding Box angewandt wobei dieser auf 256x192 pixel skalliert wird.

BILD

$ P_{j} = {p_{j}^{t-m+1}, p_{j}^{t-m+2}, ..., p_{j}^{t}} $

## Model:


## Results:

