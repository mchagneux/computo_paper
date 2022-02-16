export CUDA_VISIBLE_DEVICES=0
export IMAGES='./data/images'
export BASE_NETWORK_HEATMAPS='./data/extracted_heatmaps'
export BASE_PRETRAINED_WEIGHTS='./external_pretrained_networks/centernet_pretrained.pth'
export SYNTHETIC_VIDEOS='./data/generated_videos'
export DOWNSAMPLING_FACTOR='4'
export SYNTHETIC_OBJECTS='./data/synthetic_objects'
export VALIDATION_VIDEOS='./data/validation_videos'
export EXTERNAL_DETECTIONS='./data/detector_results'

algorithm=EKF_0
details='hungarian_internal'
experiment_name=${algorithm}_${details}
output_dir=experiments/tracking/${experiment_name}
create_clean_directory $output_dir


python src/track.py \
    --data_dir data/validation_videos/all/short_segments_12fps/videos \
    --output_dir ${output_dir} \
    --confidence_threshold 0.5 \
    --downsampling_factor ${DOWNSAMPLING_FACTOR} \
    --algorithm ${algorithm} \
    --noise_covariances_path tracking_parameters \
    --output_shape 960,544 \
    --skip_frames 0 \
    --display 1



