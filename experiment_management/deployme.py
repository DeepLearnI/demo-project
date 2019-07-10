import foundations
import numpy as np

def get_params():
    params = {
        "rnn_units": np.random.randint(256, 2049),
        "batch_size": np.random.randint(16, 256),
        "embedding_dim": np.random.randint(128, 512),
        "epochs": np.random.randint(5, 15),
        "seq_length": 100,
        "temperature": 1.,
        "dataset_url": "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
    }
    return params
    
for _ in range(3):
    foundations.deploy(
        env="scheduler",
        job_directory="experiments/text_generation_simple",
        entrypoint="driver.py",
        params=get_params(),
    )