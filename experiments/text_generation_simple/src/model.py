"""
Each function here can be considered as a step (stage) towards building a model.

Although we don't use foundations in this simple example, we import Foundations as it quickly becomes useful once want to use features like `.log_metric()` within stage functions.

We create a simple function that adds 10 to a number that will be used as a stage in our driver file.
"""

import foundations
import tensorflow as tf
import os
import time
from utils import post_slack_channel

tf.enable_eager_execution()


class Model:
    def __init__(self, vocab, embedding_dim, rnn_units, batch_size, char2idx, idx2char):
        self.vocab_size = len(vocab)
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.batch_size = batch_size
        self.char2idx = char2idx
        self.idx2char = idx2char

        if tf.test.is_gpu_available():
            self.rnn = tf.keras.layers.CuDNNGRU
        else:
            import functools
            self.rnn = functools.partial(
                tf.keras.layers.GRU, recurrent_activation='sigmoid')
            
        self.model = self.build_model()
        self.model.summary()

        def loss(labels, logits):
            return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        
        self.model.compile(
            optimizer=tf.train.AdamOptimizer(),
            loss=loss)

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim,
                                      batch_input_shape=[self.batch_size, None]),
            self.rnn(self.rnn_units,
                     return_sequences=True,
                     recurrent_initializer='glorot_uniform',
                     stateful=True),
            tf.keras.layers.Dense(self.vocab_size)
        ])
        return model
    
    def train(self, dataset, steps_per_epoch, checkpoint_dir='./training_checkpoints', epochs=30, job_id=-1):
        # Directory where the checkpoints will be saved
        checkpoint_dir = checkpoint_dir
        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

        # Define the optimizer to use
        optimizer = tf.train.AdamOptimizer()
        
        post_slack_channel('Training job starting',
                           job_id=job_id)
        
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
                    loss = tf.losses.sparse_softmax_cross_entropy(target, predictions)
        
                # Back prop
                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
                if batch_n % steps_per_epoch == 0:
                    template = '[Epoch {} - Loss {:.4f}]'
                    print(template.format(epoch + 1, loss))
                    post_slack_channel(template.format(epoch + 1, loss),
                                       job_id=job_id)
    
            # saving (checkpoint) the model every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.model.save_weights(checkpoint_prefix.format(epoch=epoch))
    
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, loss))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

            post_slack_channel('End of training: Epoch {} - Loss {}'.format(epoch + 1, loss),
                               job_id=job_id)

        self.model.save_weights(checkpoint_prefix.format(epoch=epoch))

    def test(self, dataset, steps):
        return self.model.evaluate(dataset.repeat(), steps=steps)

    def generate_text(self, start_string, checkpoint_dir='./training_checkpoints', temperature=1.):
        self.batch_size = 1
        self.model = self.build_model()
        self.model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        self.model.build(tf.TensorShape([1, None]))

        self.model.summary()
        text = self.generate(start_string, temperature=temperature)
        return text

    def generate(self, start_string, temperature=1.0):
        # Evaluation step (generating text using the learned model)
    
        # Number of characters to generate
        num_generate = 1000
    
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
        for i in range(num_generate):
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
