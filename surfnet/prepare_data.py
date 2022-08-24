import os.path as op
import os
from urllib.request import urlretrieve

def download_from_url(url, filename, folder):
    model_filename = op.realpath(op.join(folder, filename))
    os.makedirs(folder, exist_ok=True)
    if not op.exists(model_filename):
        print('---Downloading: '+filename)
        urlretrieve(url, model_filename)
    else:
        print('---Already downloaded: '+filename)
    return model_filename

def download_data(folder="data/external_detections/part_1_segment_0"):
    detection_path = "https://github.com/mchagneux/computo_paper/releases/download/data.v.1/saved_detections.pickle"
    frames_path = "https://github.com/mchagneux/computo_paper/releases/download/data.v.1/saved_frames.pickle"
    download_from_url(detection_path, "saved_detections.pickle", folder)
    download_from_url(frames_path, "saved_frames.pickle", folder)
