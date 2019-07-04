import numpy as np
from slackclient import SlackClient


slack_client = SlackClient('xoxp-6264324945-46768961556-314329155110-1f92449856f0d07a99f7598ff31e9bf2')


def get_params():
    params = {
        "rnn_units": np.random.randint(256, 2049),
        "batch_size": np.random.randint(16, 256),
        "embedding_dim": np.random.randint(128, 512),
        "epochs": 30,
        "seq_length": 100,
        "temperature": 1.,
        "dataset_url": "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
    }

    return params


def post_slack_channel(msg):
    slack_client.api_call(
        "chat.postMessage",
        channel="spamity",
        text=msg,
        username="Major_Language_Model",
        icon_emoji=":arabsteve:"
    )

