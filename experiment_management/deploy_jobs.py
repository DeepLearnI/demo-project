import numpy as np
import foundations


def get_params():
    params = {
        "rnn_layers": np.random.randint(1, 3),
        "rnn_units": np.random.randint(128, 512),
        "batch_size": np.random.randint(16, 128),
        "learning_rate": np.random.choice([0.001, 0.01, 0.0001]),
        "embedding_dim": np.random.randint(64, 256),
        "epochs": np.random.randint(3, 10),
        "seq_length": 100,
        "temperature": .1,
        "num_characters_to_generate": 200,
        "dataset_url": "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
    }
    return params

for i in range(5):
    foundations.deploy(
        env="scheduler",
        job_directory="experiments/text_generation_simple",
        entrypoint="main.py",
        params=get_params(),
    )