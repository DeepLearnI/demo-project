from utils import load_preprocessors
from model import Model
import foundations
import time


char2idx, idx2char, vocab = load_preprocessors()

params = foundations.load_parameters()

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
    generated_text = model.generate(start_string=input_text, temperature=params['temperature'], num_characters_to_generate=params['num_characters_to_generate'])
    return generated_text