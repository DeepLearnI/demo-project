import foundations as f9s

NUM_JOBS = 1

for i in range(NUM_JOBS):
    f9s.deploy(env='scheduler',
               entrypoint='new_main.py',
               job_directory='experiments/image_seg_lottery',
               project_name='marcus - segmentation')