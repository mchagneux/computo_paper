import shutil 
import os 
import random 
gt_base_dir = 'TrackEval/data/gt'
trackers_base_dir = 'TrackEval/data/trackers'

segments_name = 'surfrider_short_segments_12fps'

ratio = 0.6
seqmap_path = os.path.join(gt_base_dir, segments_name, 'seqmaps', 'surfrider-test.txt')

with open(seqmap_path, 'r') as f:
    segments = [line.strip('\n') for line in f.readlines()[1:]]


random.shuffle(segments)
segments_p1 = [segment for segment in segments if 'part_1' in segment]
segments_p2 = [segment for segment in segments if 'part_2' in segment]
segments_p3 = [segment for segment in segments if 'part_3' in segment]

val_segments = segments_p1[:int(ratio*len(segments_p1))] \
            + segments_p2[:int(ratio*len(segments_p2))] \
            + segments_p3[:int(ratio*len(segments_p3))]

test_segments = segments_p1[int(ratio*len(segments_p1)):] \
            + segments_p2[int(ratio*len(segments_p2)):] \
            + segments_p3[int(ratio*len(segments_p3)):]


for split, segments_for_split in zip(['test','val'], [test_segments, val_segments]): 
    gt_dir_split = os.path.join(gt_base_dir, f'{segments_name}_{split}')
    seqmaps_dir = os.path.join(gt_dir_split, 'seqmaps')
    os.makedirs(seqmaps_dir, exist_ok=True)
    with open(os.path.join(seqmaps_dir, 'surfrider-test.txt'), 'w') as f: 
        f.writelines(['name\n'] + [segment + '\n' for segment in segments_for_split])
    
    gt_data_dir = os.path.join(gt_base_dir, segments_name, 'surfrider-test')
    gt_data_dir_split = os.path.join(gt_dir_split, 'surfrider-test')
    os.makedirs(gt_data_dir_split, exist_ok=True)
    for data_dir in os.listdir(gt_data_dir):
        if data_dir in segments_for_split:
            shutil.copytree(os.path.join(gt_data_dir, data_dir), os.path.join(gt_data_dir_split, data_dir))


    trackers_dir = os.path.join(trackers_base_dir, segments_name, 'surfrider-test')
    trackers_dir_split = os.path.join(trackers_base_dir, f'{segments_name}_{split}', 'surfrider-test')
    for tracker_name in os.listdir(trackers_dir):
        tracker_data_dir = os.path.join(trackers_dir, tracker_name, 'data')
        tracker_data_dir_split = os.path.join(trackers_dir_split, tracker_name, 'data')
        os.makedirs(tracker_data_dir_split, exist_ok=True)
        for data_file in os.listdir(tracker_data_dir):
            if data_file.strip('.txt') in segments_for_split:
                shutil.copy(os.path.join(tracker_data_dir, data_file), os.path.join(tracker_data_dir_split, data_file))

    # os.makedirs(os.path.join(trackers_base_dir, f'{segments_name}_{split}'), exist_ok=True)
