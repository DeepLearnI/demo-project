import numpy as np
import foundations
import foundations.prototype
from slackclient import SlackClient


slack_client = SlackClient('xoxp-6264324945-46768961556-314329155110-1f92449856f0d07a99f7598ff31e9bf2')


def get_params():
    params = {
        "rnn_units": np.random.randint(256, 2049),
        "batch_size": np.random.randint(16, 256),
        "embedding_dim": np.random.randint(128, 512),
        "epochs": 1,
        "seq_length": 100,
        "temperature": 1.,
        "dataset_url": "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
    }

    for param_key, param_value in params.items():
        foundations.prototype.set_tag("_param_{}".format(param_key), param_value)

    return params


def post_slack_channel(msg, job_id):
    slack_client.api_call(
        "chat.postMessage",
        channel="space2vec-models",
        text='*%s*: %s' % (job_id, msg),
        username="Major_Language_Model",
        icon_emoji=":arabsteve:"
    )


