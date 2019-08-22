"""
Each function here can be considered as a step (stage) towards building a model.

We create a simple function that adds 10 to a number that will be used as a stage in our driver file.
"""

import foundations
import tensorflow as tf
import os
import time
from utils import post_slack_channel

tf.enable_eager_execution()


class Model:
    def __init__(self, vocab, embedding_dim, rnn_layers, rnn_units, batch_size, learning_rate, char2idx, idx2char):
        self.vocab_size = len(vocab)
        self.embedding_dim = embedding_dim
        self.rnn_layers = rnn_layers
        self.rnn_units = rnn_units
        self.batch_size = batch_size
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.loss = tf.losses.sparse_softmax_cross_entropy

        self.learning_rate = learning_rate

        if tf.test.is_gpu_available():
            print('Using GPU')
            self.rnn = tf.keras.layers.CuDNNGRU
        else:
            print('using CPU')
            import functools
            self.rnn = functools.partial(
                tf.keras.layers.GRU, recurrent_activation='sigmoid')
            
        self.model = self.build_model()
        self.model.summary()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, batch_input_shape=[self.batch_size, None], name='embedding_layer'),
            *[self.rnn(self.rnn_units,
                     return_sequences=True,
                     recurrent_initializer='glorot_uniform',
                     stateful=True, name='rnn_layer_{}'.format(i)) for i in range(self.rnn_layers)],
            tf.keras.layers.Dense(self.vocab_size, name='output_dense')
        ])
        return model

    def set_test_mode(self, checkpoint_dir):
        self.batch_size = 1
        self.model = self.build_model()
        self.load_saved_model(checkpoint_dir=checkpoint_dir)
    
    def train(self, dataset, steps_per_epoch, checkpoint_dir='./training_checkpoints', epochs=30):
        # Directory where the checkpoints will be saved
        checkpoint_dir = checkpoint_dir
        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

        # Define the optimizer to use
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        
        post_slack_channel('Training job starting')

        for epoch in range(epochs):
            start = time.time()
    
            # initializing the hidden state at the start of every epoch
            # initially hidden is None
            hidden = self.model.reset_states()
    
            for (batch_n, (inp, target)) in enumerate(dataset):
                with tf.GradientTape() as tape:
                    # feeding the hidden state back into the model
                    # This is the interesting step
                    predictions = self.model(inp)
                    loss = self.loss(target, predictions)
        
                # Back prop
                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
            template = '[Epoch {} - Loss {:.4f}]'
            print(template.format(epoch + 1, loss))
            post_slack_channel(template.format(epoch + 1, loss))

            # saving (checkpoint) the model every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.model.save_weights(checkpoint_prefix.format(epoch=epoch))
    
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

        post_slack_channel('End of training: Epoch {} - Loss {}'.format(epoch + 1, loss))

        self.model.save_weights(checkpoint_prefix.format(epoch=epoch))

    def test(self, dataset, steps):
        loss = 0
        n_batch = 0
        for n_batch, (inp, target) in enumerate(dataset):
            pred = self.model(inp)
            
            loss += self.loss(target, pred)
            
        loss /= (n_batch + 1)
    
        return float(loss)

    def load_saved_model(self, checkpoint_dir):
        self.model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    def generate(self, start_string, temperature=1.0, num_characters_to_generate=1000):
        # Converting our start string to numbers (vectorizing)
        input_eval = [self.char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)
    
        # Empty string to store our results
        text_generated = []
    
        # Low temperatures results in more predictable text.
        # Higher temperatures results in more surprising text.
        # Experiment to find the best setting.
        temperature = temperature
    
        # Here batch size == 1
        self.model.reset_states()
        for i in range(num_characters_to_generate):
            predictions = self.model(input_eval)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)
        
            # using a multinomial distribution to predict the word returned by the model
            predictions = predictions / temperature
            predicted_id = tf.multinomial(predictions, num_samples=1)[-1, 0]
        
            # We pass the predicted word as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)
        
            text_generated.append(self.idx2char[predicted_id])
    
        return start_string + ''.join(text_generated)
