import numpy as np
from slackclient import SlackClient
import pickle

slack_token = "<slack token>"
slack_client = SlackClient(slack_token)


def post_slack_channel(msg):
    slack_client.api_call(
        "chat.postMessage",
        channel="spamity",
        text=msg,
        username="Major_Language_Model",
        icon_emoji=":arabsteve:"
    )


def save_preprocessors(char2idx, idx2char, vocab):
    with open('artifacts/char2idx.pkl', 'wb') as file:
        pickle.dump(char2idx, file)

    with open('artifacts/idx2char.pkl', 'wb') as file:
        pickle.dump(idx2char, file)

    with open('artifacts/vocab.pkl', 'wb') as file:
        pickle.dump(vocab, file)

def load_preprocessors():
    with open('artifacts/char2idx.pkl', 'rb') as file:
        char2idx = pickle.load(file)

    with open('artifacts/idx2char.pkl', 'rb') as file:
        idx2char = pickle.load(file)

    with open('artifacts/vocab.pkl', 'rb') as file:
        vocab = pickle.load(file)

    return char2idx, idx2char, vocab