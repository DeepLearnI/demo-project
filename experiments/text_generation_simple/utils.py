import numpy as np
from slackclient import SlackClient
import pickle

slack_client = SlackClient('xoxp-6264324945-46768961556-314329155110-1f92449856f0d07a99f7598ff31e9bf2')


def get_params():
    params = {
        "rnn_units": 1024, #np.random.randint(256, 2049),
        "batch_size": 64, #np.random.randint(16, 256),
        "embedding_dim": 256, #np.random.randint(128, 512),
        "epochs": 1,
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

def save_preprocessors(char2idx, idx2char):
    with open('artifacts/char2idx.pkl', 'wb') as file:
        pickle.dump(char2idx, file)

    with open('artifacts/idx2char.pkl', 'wb') as file:
        pickle.dump(idx2char, file)

def load_preprocessors():
    with open('artifacts/char2idx.pkl', 'rb') as file:
        char2idx = pickle.load(file)

    with open('artifacts/idx2char.pkl', 'rb') as file:
        idx2char = pickle.load(file)

    return char2idx, idx2char