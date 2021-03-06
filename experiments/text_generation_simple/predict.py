from utils import load_preprocessors
from model import Model

char2idx, idx2char, vocab = load_preprocessors()

params = {
        "rnn_layers": 3,
        "rnn_units": 512,
        "batch_size": 64,
        "learning_rate": 0.001,
        "embedding_dim": 256,
        "epochs": 3,
        "seq_length": 100,
        "temperature": .2,
        "dataset_url": "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
    }

model = Model(vocab,
              embedding_dim=params['embedding_dim'],
              rnn_layers=params['rnn_layers'],
              rnn_units=params['rnn_units'],
              batch_size=1,
              learning_rate=params['learning_rate'],
              char2idx=char2idx,
              idx2char=idx2char)


model.load_saved_model(checkpoint_dir='./training_checkpoints')


def generate_prediction(input_text):
    generated_text = model.generate(start_string=input_text, temperature=params['temperature'], num_characters_to_generate=200)
    return generated_text
