import time
from preprocessing import download_data, preprocess_data
from model import Model
from utils import save_preprocessors, load_preprocessors


params = {
    "rnn_layers": 1,
    "rnn_units": 1024,
    "batch_size": 64,
    "learning_rate": 0.001,
    "embedding_dim": 256,
    "epochs": 3,
    "seq_length": 100,
    "temperature": 1.,
    "dataset_url": "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
}


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

test_loss = model.test(dataset_test, steps_per_epoch_test)
print("Final test loss: {}".format(test_loss))

model.set_test_mode(checkpoint_dir='./training_checkpoints')
generated_text = model.generate(start_string=u"ROMEO: ", num_characters_to_generate=25)
print("Sample generated text: \n{}".format(generated_text))
