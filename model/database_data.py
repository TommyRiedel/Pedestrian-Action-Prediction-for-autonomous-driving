"""Extract relevant informations (BBox, pose, ego-vehicle speed, ...) of sequence from JAAD / PIE sequence 
and saves them as pickle files (organized based on dataset type, data split and the specific seq.)
"""
import pickle
from jaad_data import JAAD
from pie_data import PIE
import numpy as np

# Choose the dataset (JAAD or PIE)
data_which = "pie"

# Set data/subset type (train, val or test)
# and specify the path to the dataset
# and generate data sequence with function in (jaad_data, pie_data)

if data_which == "jaad":
    data_type = "val"
    path = "/home/ubuntu/thomas_ma/JAAD/"
    imdb = JAAD(data_path=path)
    seq = imdb.generate_data_trajectory_sequence(
        image_set=data_type, seq_type="crossing", sample_type="beh"
    )
elif data_which == "pie":
    data_type = "val"
    path = "/home/ubuntu/thomas_ma/PIE/"
    imdb = PIE(data_path=path)
    seq = imdb.generate_data_trajectory_sequence(
        image_set=data_type, seq_type="crossing"
    )

# Set some parameters (action prediction benchmark): TTE = [30, 60] frames; observation length = 16 frames
time_to_event_max = 60
time_to_event_min = 30
obs_length = 16
pos_count = 0
neg_count = 0
k = 0

# Initialize a dictionary to store data:
# ped. ID, image width, seq. of frames, time-to-event, seq. of bbox, seq. of pose,
# seq. of bbox center, seq. of images, ped. actions, seq. of ego-vehicle velocity
data = {
    "pid": "",
    "width": None,
    "frames": [],
    "tte": None,
    "bbox": [],
    "pose": [],
    "center": [],
    "images": [],
    "activities": None,
    "vel": [],
}  ### veh_speed?

for pid in range(0, len(seq["bbox"])):
    # Define the # of of the first frames of the seq. (TTE = 60 frames) and the seq. (TTE = 30 frames)
    start_idx = len(seq["bbox"][pid]) - obs_length - time_to_event_max
    end_idx = len(seq["bbox"][pid]) - obs_length - time_to_event_min
    # Set overlap based on dataset (JAAD, PIE) and data/subset type (train, val, test) and calculate step size:
    # PIE + train => overlap of 60% == every 6th frame
    # JAAD + train => overlap of 80% == every 3th frame
    # else => overlap = 0%!
    if (data_type == "train") and (data_which == "pie"):
        overlap = 0.6
    elif (data_type == "train") and (data_which == "jaad"):
        overlap = 0.8
    else:
        overlap = 0
    step_size = obs_length if overlap == 0 else round((1 - overlap) * obs_length)
    step_size = 1 if step_size < 1 else step_size
    # Loop trough the sequence with the specified step size
    for i in range(start_idx, end_idx + 1, step_size):
        if i >= 0:
            # Populate the data dictionary with relevant information from the loaded sequence
            data["pid"] = seq["pid"][pid][0][0]
            if data_which == "pie":
                data["width"] = 1920
            elif data_which == "jaad":
                data["width"] = seq["image_dimension"][0]
            data["frames"] = [x for x in range(i, i + 16)]
            data["tte"] = len(seq["bbox"][pid]) - (i + obs_length)
            # Extract pose and bbox information for each frame in the sequence
            pose = [[] for _ in range(16)]
            bbox = [[] for _ in range(16)]
            for x in range(0, 16):
                buffer1 = np.array((seq["pose"][pid][x]))
                buffer2 = np.array((seq["bbox"][pid][x]))
                while buffer1.size > 0:
                    pose[x].append(buffer1[:2])
                    buffer1 = buffer1[2:]
                while buffer2.size > 0:
                    bbox[x].append(buffer2[:2])
                    buffer2 = buffer2[2:]
            data["pose"] = np.array(pose)
            data["bbox"] = np.array(bbox)
            # data['center'] = np.array([seq['center'][pid][x] for x in range(i, i+16)])
            data["images"] = [seq["image"][pid][x] for x in range(i, i + 16)]
            data["activities"] = seq["activities"][pid][i][0]
            # Include vehicle speed information only for the 'pie' dataset
            if data_which == "pie":
                data["vel"] = np.array(
                    [seq["obd_speed"][pid][x] for x in range(i, i + 16)]
                )
            # Update positive and negative count based on activity label
            if data["activities"] == 1:
                pos_count += 1
            elif data["activities"] == 0:
                neg_count += 1
            print(
                "Negative {} and positive {} sample counts".format(neg_count, pos_count)
            )
            # Save the data dictionary as a pickle file
            if data_which == "pie":
                filename = (
                    "data/pie/"
                    + data_type
                    + "/data_pid_"
                    + data["pid"]
                    + "_fr_"
                    + str(i)
                    + ".pickle"
                )
            elif data_which == "jaad":
                filename = (
                    "data/jaad/"
                    + data_type
                    + "/data_pid_"
                    + data["pid"]
                    + "_fr_"
                    + str(i)
                    + ".pickle"
                )
            with open(filename, "wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
