from concurrent.futures import process
import os 
import subprocess 


kappa_values = [1,3,5,7]
tau_values = [i for i in range(1,10)]
method_name = 'fairmot'

base_dir = 'TrackEval/data/trackers/surfrider_short_segments_12fps/surfrider-test'
processes = []
for kappa in [1,3,5,7]:
    for tau in [i for i in range(1,10)]:
        if not (kappa == 1 and tau == 0):
            results_dir = os.path.join(base_dir, f'{method_name}_kappa_1_tau_0', 'data')
            processed_results_dir = os.path.join(base_dir, f'{method_name}_kappa_{kappa}_tau_{tau}', 'data')
            os.makedirs(processed_results_dir, exist_ok=True)
            for filename in os.listdir(results_dir):
                input_file = os.path.join(results_dir, filename)
                output_name = os.path.join(processed_results_dir, filename)
                processes.append(subprocess.Popen(f'python surfnet/postprocess_and_count_tracks.py \
                                    --input_file {input_file} \
                                    --output_name {output_name} \
                                    --kappa {kappa} \
                                    --tau {tau}',
                                    shell=True))


for p in processes:
    p.wait()

