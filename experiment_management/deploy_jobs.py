import numpy as np
import foundations

NUM_JOBS = 1

def get_params():
    params = {
        "rnn_layers": np.random.randint(1, 4),
        "rnn_units": int(np.random.choice([128, 256, 512])),
        "batch_size": int(np.random.choice([32, 64, 128, 256])),
        "learning_rate": np.random.choice([0.001, 0.01, 0.005]),
        "embedding_dim": np.random.randint(128, 257),
        "epochs": np.random.randint(5, 16),
        "seq_length": 100,
        "temperature": np.random.choice([.2, .3, .4]),
        "num_characters_to_generate": 200,
        "dataset_url": "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
    }
    return params

for i in range(NUM_JOBS):
    foundations.deploy(
        env="scheduler",
        job_directory="experiments/text_generation_simple",
        entrypoint="main.py",
        project_name="some_project_name",
        params=get_params(),
    )