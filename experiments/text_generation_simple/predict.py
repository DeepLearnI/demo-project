from utils import load_preprocessors
from model import Model
import foundations


char2idx, idx2char, vocab = load_preprocessors()

params = foundations.load_parameters()

model = Model(vocab,
              embedding_dim=params['embedding_dim'],
              rnn_units=params['rnn_units'],
              batch_size=1,
              char2idx=char2idx,
              idx2char=idx2char)

model.load_saved_model(checkpoint_dir='./training_checkpoints')

def predict(input_text):
    generated_text = model.generate(start_string=input_text, temperature=params['temperature'])
    return generated_text