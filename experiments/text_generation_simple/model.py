"""
Each function here can be considered as a step (stage) towards building a model.

We create a simple function that adds 10 to a number that will be used as a stage in our driver file.
"""

import tensorflow as tf
import os
import time
from utils import post_slack_channel

# tf.enable_eager_execution()


class Model:
    def __init__(self, vocab, embedding_dim, rnn_units, batch_size, char2idx, idx2char):
        self.vocab_size = len(vocab)
        self.embedding_dim = embedding_dim
        self.rnn_layers = rnn_layers
        self.rnn_units = rnn_units
        self.batch_size = batch_size
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.loss_func = tf.losses.sparse_softmax_cross_entropy
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate)

        if tf.test.is_gpu_available():
            print('Using GPU')
            self.rnn = tf.keras.layers.CuDNNGRU
        else:
            print('using CPU')
            import functools
            self.rnn = functools.partial(
                tf.keras.layers.GRU, recurrent_activation='sigmoid')

        self.inputs = tf.Placeholder(tf.int32, shape=(None, None), name='inputs')
        self.targets = tf.Placeholder(tf.float32, shape=(None, None), name='targets')

        self.logits = self.forward_prop(self.inputs)
        self.compute_loss()
        self.back_prop()
        self.generation_loop()

        self.session = tf.Session()
        self.saver = tf.train.Saver()

    def forward_prop(self, inputs):
        character_embeddings = tf.get_variable('character_embeddings', [self.vocab_size, self.embedding_dim],
                                               dtype=tf.float32)
        x = tf.nn.embedding_lookup(character_embeddings, self.inputs)

        self.rnn_stack = tf.keras.layers.RNN(tf.contrib.cudnn_rnn.CudnnGRU(self.rnn_layers, self.rnn_units),
                                             return_sequences=True, unroll=True)
        x = self.rnn_stack(x)

        x = tf.layers.dense(x, self.vocab_size)

        logits = x

        self.trainable_variables = tf.trainable_variables()

    def compute_loss(self):
        with tf.GradientTape() as self.tape:
            self.loss = self.loss_func(self.targets, self.logits)

    def reset_states(self):
        self.rnn_stack.reset_states()

    def back_prop(self):
        # Back prop
        grads = self.tape.gradient(self.loss, self.trainable_variables)
        self.optimize = self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

    def save_model(self, checkpoint_path):
        self.saver.save(self.session, checkpoint_path)

    def load_saved_model(self, checkpoint_path):
        checkpoint_state = tf.train.get_checkpoint_state(checkpoint_path)
                    
        if checkpoint_state and checkpoint_state.model_checkpoint_path:
            print('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path))
            self.saver.restore(self.session, checkpoint_state.model_checkpoint_path)
        
        else:
            print('No model to load at {}'.format(checkpoint_path))
            self.saver.save(self.session, checkpoint_path)
    
    def train(self, dataset, steps_per_epoch, checkpoint_dir='./training_checkpoints', epochs=30):
        # Directory where the checkpoints will be saved
        checkpoint_dir = checkpoint_dir
        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
        
        post_slack_channel('Training job starting')

        for epoch in range(epochs):
            start = time.time()

            self.reset_states()
            for inp, tar in dataset:
                _, loss = self.session.run([self.optimize, self.loss], feed_dict={self.inputs: inp.astype(np.int32),
                                                                                  self.targets: tar.astype(np.float32)})
            
        
            template = '[Epoch {} - Loss {:.4f}]'
            print(template.format(epoch + 1, loss))
            post_slack_channel(template.format(epoch + 1, loss))

            # saving (checkpoint) the model every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_model(checkpoint_prefix.format(epoch=epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

        post_slack_channel('End of training: Epoch {} - Loss {}'.format(epoch + 1, loss))

        self.save_model(checkpoint_prefix.format(epoch=epoch))

    def test(self, dataset, steps):
        loss = 0
        n_batch = 0
        for n_batch, (inp, target) in enumerate(dataset):
            l = self.session(self.loss, feed_dict={self.inputs: inp.astype(np.int32), self.targets: target.astype(np.float32)})
            loss += l
            
        loss /= (n_batch + 1)
    
        return float(loss)

    def generation_loop(self):
        # Here batch size == 1
        self.reset_states()
        initial_time = tf.constant(0, dtype=tf.int32)
        self.initial_input_eval = tf.Placeholder(tf.int32, shape=(1, None), name='initial_text')
        self.initial_generated_text = tf.Placeholder(tf.int32, shape=(None, ), name='initial_generated_text_as_int')

        def condition(time, unused_input_eval, unused_generated_text):
            return tf.less_equal(time, self.num_generate)

        def body(time, input_eval, generated_text):
            predictions = self.forward_prop(input_eval)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)
        
            # using a multinomial distribution to predict the word returned by the model
            predictions = predictions / self.temperature
            predicted_id = tf.multinomial(predictions, num_samples=1)[-1, 0]
        
            # We pass the predicted word as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)

            generated_text = tf.concat([generated_text, predicted_id], axis=0)

            time = time + 1
            return time, input_eval, generated_text

        _, _, generated_text = tf.while_loop(
            condition,
            body,
            loop_vars=[
                initial_time, initial_input_eval, initial_generated_text
            ],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([1, None]),
                tf.TensorShape([None])
            ],
            back_prop=False,
            parallel_iterations=32,
            swap_memory=False
        )
        self.generated_text = generated_text

    def generate(self, start_string, temperature=1.0):
        # Evaluation step (generating text using the learned model)
        #     
        # Converting our start string to numbers (vectorizing)
        input_eval = [self.char2idx[s] for s in start_string]
     
        text_generated = self.session.run(self.generated_text, 
                                          feed_dict={self.initial_input_eval: np.expand_dims(input_eval, 0).astype(np.int32),
                                                     self.initial_generated_text: input_eval.astype(np.int32)})

        text_generated = [self.idx2char[s] for s in text_generated]
        return ''.join(text_generated)
