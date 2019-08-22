import foundations as f9s

NUM_JOBS = 3

for i in range(NUM_JOBS):
    f9s.deploy(env='scheduler',
               entrypoint='new_main.py',
               project_name='marcus - CC fraud')