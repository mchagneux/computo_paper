import os
import shutil 


trackers_paths = 'TrackEval/data/trackers/surfrider_short_segments_12fps/surfrider-test'

tracker_base_name = 'ours_EKF_1'

for tracker_version in os.listdir(trackers_paths):
    if tracker_base_name in tracker_version: 
        tau = tracker_version.split('_')[-1]
        if 'v0' in tracker_version: 
            kappa = 1
        else:
            kappa = tracker_version.split('_')[5]
        shutil.move(os.path.join(trackers_paths, tracker_version), 
                    os.path.join(trackers_paths, f'{tracker_base_name}_kappa_{kappa}_tau_{tau}'))
        