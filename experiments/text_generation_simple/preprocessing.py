import tensorflow as tf
import numpy as np


def download_data(remote_path, local_path):
    path_to_file = tf.keras.utils.get_file(local_path, remote_path)
    return path_to_file


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def preprocess_data(path_to_file, params):
    # Read, then decode for py2 compat.
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    
    # Get the unique characters in the file (vocab)
    vocab = sorted(set(text))
    
    # Creating a mapping from unique characters to indices
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    
    # Convert text to indices
    text_as_int = np.array([char2idx[c] for c in text])
    
    # The maximum length sentence we want for a single input in characters
    seq_length = params['seq_length']
    
    text_as_int_train, text_as_int_test = text_as_int[:int(len(text_as_int) * 0.9)], text_as_int[int(-0.1 * len(text_as_int)):]
    
    examples_per_epoch_train = len(text_as_int_train) // seq_length
    examples_per_epoch_test = len(text_as_int_test) // seq_length
    
    # Create training examples / targets
    char_dataset_train = tf.data.Dataset.from_tensor_slices(text_as_int_train)
    
    # Create validation examples / targets
    char_dataset_test = tf.data.Dataset.from_tensor_slices(text_as_int_test)
    
    sequences_train = char_dataset_train.batch(seq_length + 1, drop_remainder=True)
    sequences_test = char_dataset_test.batch(seq_length + 1, drop_remainder=True)
    
    dataset_train = sequences_train.map(split_input_target)
    dataset_test = sequences_test.map(split_input_target)
    
    # Batch size
    batch_size = params['batch_size']
    steps_per_epoch_train = examples_per_epoch_train // batch_size
    steps_per_epoch_test = examples_per_epoch_test // batch_size
    
    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 10000
    
    dataset_train = dataset_train.shuffle(BUFFER_SIZE).batch(batch_size, drop_remainder=True)
    dataset_test = dataset_test.shuffle(BUFFER_SIZE).batch(batch_size, drop_remainder=True)
    
    return dataset_train, dataset_test, steps_per_epoch_train, steps_per_epoch_test, vocab, char2idx, idx2char