import time
from preprocessing import download_data, preprocess_data
from model import Model
from utils import save_preprocessors, load_preprocessors
import numpy as np
import foundations


params = foundations.load_parameters()

path_to_file = download_data('https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt', 'shakespeare.txt')

dataset_train, dataset_test, steps_per_epoch_train, steps_per_epoch_test, vocab, char2idx, idx2char = preprocess_data(path_to_file, params)

# Save preprocessors for serving
save_preprocessors(char2idx, idx2char, vocab)

model = Model(vocab,
              embedding_dim=params['embedding_dim'],
              rnn_layers=params['rnn_layers'],
              rnn_units=params['rnn_units'],
              batch_size=params['batch_size'],
              learning_rate=params['learning_rate'],
              char2idx=char2idx,
              idx2char=idx2char)

model.train(dataset_train,
            steps_per_epoch=steps_per_epoch_train,
            checkpoint_dir='./training_checkpoints',
            epochs=params['epochs'],)

train_loss = model.test(dataset_train, steps_per_epoch_train)
print("Final train loss: {}".format(train_loss))
foundations.log_metric('train_loss', train_loss)

test_loss = model.test(dataset_test, steps_per_epoch_test)
print("Final test loss: {}".format(test_loss))
foundations.log_metric('test_loss', test_loss)

model.set_test_mode(checkpoint_dir='./training_checkpoints')

start_time = time.time()
generated_text = model.generate(
    start_string=u"ROMEO: ",
    temperature=params['temperature'],
    num_characters_to_generate=params['num_characters_to_generate'],
)
print('synthesis time: {}'.format(time.time() - start_time))
print("Sample generated text: \n{}".format(generated_text))
foundations.log_metric('generated_text', generated_text[:50])
