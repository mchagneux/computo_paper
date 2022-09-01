fps=12
segments=short
python scripts/run_mot_challenge.py \
    --GT_FOLDER data/gt/surfrider_${segments}_segments_${fps}fps_val \
    --TRACKERS_FOLDER data/trackers/surfrider_${segments}_segments_${fps}fps_val \
    --DO_PREPROC False \
    --USE_PARALLEL True \
    --METRICS HOTA
