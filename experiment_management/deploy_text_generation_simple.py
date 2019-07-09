import foundations



foundations.deploy(env="scheduler", job_directory="/opt/foundations/cole-test/demo-project/experiments/text_generation_simple", entrypoint="src/driver.py", project_name="cole-serving")