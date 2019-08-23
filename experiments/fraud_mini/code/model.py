"""
Copyright (C) DeepLearning Financial Technologies Inc. - All Rights Reserved
Unauthorized copying, distribution, reproduction, publication, use of this file, via any medium is strictly prohibited.
Proprietary and confidential – June 2019
"""

import foundations
import os
import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
import time
import traceback

from zoneout_lstm import ZoneoutLSTMCell

from tqdm import tqdm
from capture_rate import evaluate_and_plot
from projection import Projection
from masked_loss import MaskedSigmoidCrossEntropy, MaskedFPPenalty, sequence_mask
from utils import time_string, ValueWindow

import tensorflow.contrib.tensorrt as trt
from tensorflow.python.platform import gfile



class SequentialModel:
    def __init__(self, all_params):
        '''Basic constructor to hold model parameters (supposed to be under tf.args instance)
        Args:
            - args: tf.args instance that holds different model parameters (n_layers, units, etc)
        '''
        self._args = all_params
    
    def initialize(self, data, input_lengths, is_training=False, is_evaluating=True):
        '''Initilize tensorflow graph holding the model logic (both train and test)
        Args:
            - inputs: tf placeholder with the model inputs (supposedly formatted as timeseries)
            - cat_inputs: tf placeholder with the model categorical inputs (supposedly formatted as timeseries)
            - input_lengths: tf placeholder specifying the lengths of inputs series.
            - targets: binary classification target in tf placeholder.
        '''
        with tf.variable_scope('inference'):
            inputs = data[:, :, :-1]
            targets = data[:, :, -1:]
            
            hp = self._args

            if hp.recurrency_type == 'LSTM':
                # Initalize loop layers
                lstm_cells = []
                for i, units in enumerate(hp.lstm_units):
                    lstm_cells.append(ZoneoutLSTMCell(units, is_training,
                                                      zoneout_factor_cell=hp.zoneout[i], zoneout_factor_output=hp.zoneout[i],
                                                      name='LSTM_{}'.format(i)))
                
                for cell in lstm_cells:
                    inputs, _ = tf.nn.dynamic_rnn(
                        cell,
                        inputs,
                        sequence_length=input_lengths,
                        dtype=tf.float32,
                        swap_memory=False)

            else:
                assert hp.recurrency_type == 'MLP'
            
            # During Training, project all timesteps (Previous timesteps, independently from how long previous context is are used for model training)
            dense_layers = []
            for i, units in enumerate(hp.dense_units):
                dense_layers.append(Projection(is_training, drop_rate=hp.dropout[i], shape=units, activation=tf.nn.relu,
                                               using_selu=hp.hidden_activation == 'selu', scope='hidden_dense_{}'.format(i)))
            
            for layer in dense_layers:
                inputs = layer(inputs)
            
            # No dropout for last layer
            projection_layer = Projection(is_training, shape=1, activation=None, scope='projection_layer', using_selu=False)
            logits = projection_layer(inputs)
            outputs = tf.nn.sigmoid(logits)
            
            self.input_lengths = input_lengths  # input/target lengths
            self.logits = logits
            self.outputs = outputs  # Output Probabilities after sigmoid application
            
            if self._args.label_smoothing:
                targets = tf.where(tf.equal(targets, 0), targets + self._args.negative_smoothing, targets - self._args.positive_smoothing)
            
            self.targets = targets  # Binary targets
            
            if hp.recurrency_type == 'TCN':
                print('TCN units: {}'.format(hp.tcn_units))
                print('Dense units: {}'.format(hp.dense_units))
                print('Kernels: {}'.format(hp.kernel_sizes))
                print('Dilations: {}'.format(hp.dilations))
                print('Receptive Field: {}'.format(self.receptive_field_size()))
            
            self.all_vars = tf.trainable_variables()
            print('Model Parameters: {:.3f} Million'.format(np.sum([np.prod(v.get_shape().as_list()) for v in self.all_vars]) / 1000000))
            
            # Some logs
            print('Model train mode: {}'.format(is_training))
            print('Model eval mode: {}'.format(is_evaluating))
            
            if hp.use_ema:
                self.ema = tf.train.ExponentialMovingAverage(decay=hp.ema_decay)
    
    def receptive_field_size(self):
        receptive_field = sum([(k - 1) * d for k, d in zip(self._args.kernel_sizes, self._args.dilations)])
        if self._args.recurrency_type == 'TCN':
            receptive_field *= 2
            return receptive_field + 1
        
        # Else, no temporal access used (MLP) ==> No receptive field
        return 0
    
    def add_loss(self):
        '''Adds loss functions to the graph (used to train the model)'''
        with tf.variable_scope('loss'):
            hp = self._args
            
            mask = sequence_mask(self.input_lengths, max_len=tf.shape(self.targets)[1], expand=True)
            
            self.regularization = tf.reduce_mean([tf.nn.l2_loss(v) for v in self.all_vars if not ('bias' in v.name
                                                                                                  or 'LSTM' in v.name or 'projection_' in v.name or 'TCN_' in v.name or 'LayerNorm' in v.name or 'embedding_' in v.name
                                                                                                  or 'batch_normalization' in v.name)]) * hp.reg_weight
            self.cross_entropy_loss = MaskedSigmoidCrossEntropy(self.targets, self.logits, self.input_lengths, self._args, mask=mask)
            
            self.loss = self.cross_entropy_loss + self.regularization
    
    def add_optimizer(self, global_step):
        '''Adds a training optimizer'''
        with tf.variable_scope('optimizer'):
            hp = self._args
            
            self.learning_rate = self._learning_rate_decay(global_step)
            
            optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.beta1, hp.beta2, hp.epsilon)
            
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            self.gradients = gradients
            
            # Gradient clipping if needed
            if hp.clip_gradients:
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, hp.grad_clip_val)
            else:
                clipped_gradients = gradients
            
            # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                adam_optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
                                                          global_step=global_step)
            
            if self._args.use_ema:
                # Add exponential moving average on model parameters
                # Use optimisation process as dependecy
                with tf.control_dependencies([adam_optimize]):
                    # Create the shadow variables and add ops to mainting moving average
                    # also update moving averages after each update step
                    # This is the optimize call instead of traditional adam_optimize one>
                    assert set(self.all_vars) == set(variables)
                    self.optimize = self.ema.apply(variables)
            
            else:
                self.optimize = adam_optimize
    
    def _learning_rate_decay(self, global_step):
        hp = self._args
        
        lr = tf.train.exponential_decay(hp.learning_rate,
                                        global_step,
                                        hp.decay_steps,
                                        hp.decay_rate,
                                        name='lr_exponential_decay')
        
        return tf.maximum(lr, hp.min_learning_rate)


class Model:
    def __init__(self, log_dir, all_params):
        self.all_params = all_params
        self.save_dir = os.path.join(log_dir, 'pretrained')
        self.eval_dir = os.path.join(log_dir, 'eval-dir')
        self.meta_dir = os.path.join(log_dir, 'metas')
        self.tensorboard_dir = os.path.join(log_dir, 'events')
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.save_dir, 'Sequential_model.ckpt')
        
        print('Checkpoint path: {}'.format(self.checkpoint_path))
        print('Using model: {}'.format(self.all_params.model))
        
        # Start by setting a seed for repeatability
        tf.set_random_seed(self.all_params.random_seed)
        
        # Set up data feeder
        self.coord = tf.train.Coordinator()
        
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
    def model_train_mode(self, data, data_lengths, global_step):
        with tf.variable_scope('Sequential_model', reuse=tf.AUTO_REUSE):
            model = SequentialModel(self.all_params)
            model.initialize(data, data_lengths, is_training=True, is_evaluating=False)
            model.add_loss()
            model.add_optimizer(global_step)
            stats = add_train_stats(model)
            return model, stats
    
    def model_eval_mode(self, data, data_lengths):
        with tf.variable_scope('Sequential_model', reuse=tf.AUTO_REUSE):
            model = SequentialModel(self.all_params)
            model.initialize(data, data_lengths, is_training=False, is_evaluating=True)
            model.add_loss()
            return model
    
    def train(self, dataset_train, dataset_test, dataset_train_lengths, dataset_test_lengths):
        
        # Setup data loaders
        
        with tf.variable_scope('train_iterator'):
            self.iterator_data_train = dataset_train.make_initializable_iterator()
            self.iterator_length_train = dataset_train_lengths.make_initializable_iterator()
            next_train_data = self.iterator_data_train.get_next()
            next_train_length = self.iterator_length_train.get_next()
        
        with tf.variable_scope('test_iterator'):
            self.iterator_data_test = dataset_test.make_initializable_iterator()
            self.iterator_length_test = dataset_test_lengths.make_initializable_iterator()
            next_test_data = self.iterator_data_test.get_next()
            next_test_length = self.iterator_length_test.get_next()
        
        # Set up model
        self.initializers = [i.initializer for i in
                             [self.iterator_data_test, self.iterator_data_train, self.iterator_length_test, self.iterator_length_train]]
        self.model, self.stats = self.model_train_mode(next_train_data,
                                                       next_train_length, self.global_step)
        self.eval_model = self.model_eval_mode(next_test_data, next_test_length)
        
        if self.all_params.use_ema:
            self.saver = create_shadow_saver(self.model, self.global_step)
        
        else:
            self.saver = tf.train.Saver(max_to_keep=100)
        
        # Book keeping
        step = 0
        time_window = ValueWindow(100)
        loss_window = ValueWindow(100)
        
        print('Training set to a maximum of {} steps'.format(self.all_params.train_steps)) \
            # Memory allocation on the GPU as needed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        # Train
        print('Starting training')
        with tf.Session(config=config) as sess:
            summary_writer = tf.summary.FileWriter(self.tensorboard_dir, sess.graph)
            # Allow the full trace to be stored at run time.
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # Create a fresh metadata object:
            run_metadata = tf.RunMetadata()
            
            sess.run(tf.global_variables_initializer())
            for init in self.initializers:
                sess.run(init)
            
            # saved model restoring
            if self.all_params.restore:
                # Restore saved model if the user requested it, default = True
                try:
                    checkpoint_state = tf.train.get_checkpoint_state(self.save_dir)
                    
                    if (checkpoint_state and checkpoint_state.model_checkpoint_path):
                        print('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path))
                        self.saver.restore(sess, checkpoint_state.model_checkpoint_path)
                    
                    else:
                        print('No model to load at {}'.format(self.save_dir))
                        self.saver.save(sess, self.checkpoint_path, global_step=self.global_step)
                
                except tf.errors.OutOfRangeError as e:
                    print('Cannot restore checkpoint: {}'.format(e))
            else:
                print('Starting new training!')
                self.saver.save(sess, self.checkpoint_path, global_step=self.global_step)
            
            # Training loop
            while not self.coord.should_stop() and step < self.all_params.train_steps:
                start_time = time.time()
                step, loss, opt = sess.run([self.global_step, self.model.loss, self.model.optimize], options=options, run_metadata=run_metadata)
                time_window.append(time.time() - start_time)
                loss_window.append(loss)
                message = 'Step {:7d} [{:.3f} sec/step, loss={:.5f}, avg_loss={:.5f}]'.format(
                    step, time_window.average, loss, loss_window.average)
                print(message)
                
                if np.isnan(loss) or loss > 100.:
                    print('Loss exploded to {:.5f} at step {}'.format(loss, step))
                    raise Exception('Loss exploded')
                
                if step % self.all_params.summary_interval == 0:
                    print('\nWriting summary at step {}'.format(step))
                    foundations.log_metric('training_loss', loss, step)
                    summary_writer.add_summary(sess.run(self.stats), step)
                
                if step % self.all_params.checkpoint_interval == 0 or step == self.all_params.train_steps:
                    print('Saving model!')
                    # Save model and current global step
                    self.saver.save(sess, self.checkpoint_path, global_step=self.global_step)
                
                if step % self.all_params.eval_interval == 0:
                    # Run eval and save eval stats
                    print('\nRunning evaluation at step {}'.format(step))
                    
                    all_logits = []
                    all_outputs = []
                    all_targets = []
                    all_lengths = []
                    val_losses = []
                    
                    for i in tqdm(range(4)):
                        val_loss, logits, outputs, targets, lengths = sess.run([self.eval_model.loss,
                                                                                self.eval_model.logits,
                                                                                self.eval_model.outputs,
                                                                                self.eval_model.targets,
                                                                                self.eval_model.input_lengths])
                        
                        all_logits.append(logits)
                        all_outputs.append(outputs)
                        all_targets.append(targets)
                        all_lengths.append(lengths)
                        val_losses.append(val_loss)
                    
                    logits = [l for logits in all_logits for l in logits]
                    outputs = [o for output in all_outputs for o in output]
                    targets = [t for target in all_targets for t in target]
                    lengths = [l for length in all_lengths for l in length]
                    
                    logits = np.array([e for o, l in zip(logits, lengths) for e in o[:l]]).reshape(-1)
                    outputs = np.array([e for o, l in zip(outputs, lengths) for e in o[:l]]).reshape(-1)
                    targets = np.array([e for t, l in zip(targets, lengths) for e in t[:l]]).reshape(-1)
                    
                    val_loss = sum(val_losses) / len(val_losses)
                    
                    foundations.log_metric('validation_loss', val_loss, step=step)

                    assert len(targets) == len(outputs)
                    capture_rate, fig_path = evaluate_and_plot(outputs, targets, index=np.arange(0, len(targets)),
                                                               model_name=self.all_params.name or self.all_params.self.model,
                                                               weight=self.all_params.capture_weight, out_dir=self.eval_dir, use_tf=False, sess=sess,
                                                               step=step)
                    
                    foundations.log_metric('validation_capture_rate', capture_rate, step=step)
                    add_eval_stats(summary_writer, step, val_loss, capture_rate)
                    
                    tensorboard_file = os.path.join(self.tensorboard_dir, os.listdir(self.tensorboard_dir)[0])
                    foundations.save_artifact(tensorboard_file, "tensorboard")
                    
                    ###### Replace these lines ###########################
                    print(f'train_loss:  {float(loss)}')
                    print(f'validation_loss{float(val_loss)}')
                    print(f'validation_capture_rate{float(capture_rate)}')
                    
                    ######################################################
                
            print('Training complete after {} global steps!'.format(self.all_params.train_steps))
    
    def predict(self, dataset_test, dataset_test_lengths):
        # tf.reset_default_graph()
        # Setup data loaders
        with tf.variable_scope('test_iterator'):
            self.iterator_data_test = dataset_test.make_initializable_iterator()
            self.iterator_length_test = dataset_test_lengths.make_initializable_iterator()
            next_test_data = self.iterator_data_test.get_next()
            next_test_length = self.iterator_length_test.get_next()
        # Set up model
        self.initializers = [i.initializer for i in
                             [self.iterator_data_test, self.iterator_length_test]]
        self.eval_model = self.model_eval_mode(next_test_data, next_test_length)
        
        self.saver = tf.train.Saver()
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for init in self.initializers:
                sess.run(init)
            checkpoint_state = tf.train.get_checkpoint_state(self.save_dir)
            print('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path))
            self.saver.restore(sess, checkpoint_state.model_checkpoint_path)
            time_list = []
            for i in range(10):
                start_time = time.time()
                outputs = sess.run([self.eval_model.outputs])
                elapsed_time = time.time() - start_time
                if i > 0:
                    time_list.append(elapsed_time)
            print(f'average time taken for inference per data is  {np.mean(time_list)}')


# TODO: DELTE THIS LATER
def add_train_stats(model):
    with tf.variable_scope('stats') as scope:
        tf.summary.scalar('regularization_loss', model.regularization)
        tf.summary.scalar('cross_entropy_loss', model.cross_entropy_loss)
        tf.summary.scalar('loss', model.loss)
        tf.summary.scalar('learning_rate', model.learning_rate)  # Control learning rate decay speed
        
        tf.summary.histogram('model_logits', model.logits)
        tf.summary.histogram('model_outputs', model.outputs)
        tf.summary.histogram('model_targets', model.targets)
        
        for var in model.all_vars:
            tf.summary.histogram(var.name, var)
        
        gradient_norms = [tf.norm(grad) for grad in model.gradients if
                          grad is not None]  # handle None grads (final GLU residual layer when using skips has a None grad because its output is not used)
        tf.summary.histogram('gradient_norm', gradient_norms)
        tf.summary.scalar('max_gradient_norm',
                          tf.reduce_max(gradient_norms))  # visualize gradients (in case of explosion)
        
        return tf.summary.merge_all()


def add_eval_stats(summary_writer, step, loss, capture_rate):
    values = [
        tf.Summary.Value(tag='Sequential_eval_model/eval_stats/val_loss', simple_value=loss),
        tf.Summary.Value(tag='Sequential_eval_model/eval_stats/eval_capture_rate', simple_value=capture_rate), ]
    
    test_summary = tf.Summary(value=values)
    summary_writer.add_summary(test_summary, step)


def create_shadow_saver(model, global_step=None, loading=False):
    '''Load shadow variables of saved model.
    Inspired by: https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    Can also use: shadow_dict = model.ema.veriables_to_restore()
    '''
    # Add global step to saved variables to save checkpoints correctly
    shadow_variables = [model.ema.average_name(v) for v in model.all_vars]
    variables = model.all_vars
    
    if loading:
        shadow_variables = ['Sequential_model/optimizer/' + s for s in shadow_variables]
    
    if global_step is not None:
        shadow_variables += ['global_step']
        variables += [global_step]
    
    shadow_dict = dict(zip(shadow_variables, variables))  # dict(zip(keys, values)) = {key1: value1, key2: value2, ...}
    return tf.train.Saver(shadow_dict, max_to_keep=100)


def save_freeze_tensorflow_model_for_inference(log_dir):
    save_dir = os.path.join(log_dir, 'pretrained')
    # has to be use this setting to make a session for TensorRT optimization
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.50))) as sess:
        checkpoint_state = tf.train.get_checkpoint_state(save_dir)
        saver = tf.train.import_meta_graph(f"{checkpoint_state.model_checkpoint_path}.meta")
        saver.restore(sess, checkpoint_state.model_checkpoint_path)
        your_outputs = ["Sequential_model_1/inference/Sigmoid"]
        # convert to frozen model
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess,  # session
            tf.get_default_graph().as_graph_def(),  # graph+weight from the session
            output_node_names=your_outputs)
        # write the TensorRT model to be used later for inference
        with gfile.FastGFile(os.path.join(save_dir, "frozen_model.pb"), 'wb') as f:
            f.write(frozen_graph.SerializeToString())
        print("Frozen model is successfully stored!")
        return frozen_graph, your_outputs


def convert_to_tensor_rt(log_dir, frozen_graph, your_outputs):
    save_dir = os.path.join(log_dir, 'pretrained')
    # convert (optimize) frozen model to TensorRT model
    trt_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph,  # frozen model
        outputs=your_outputs,
        max_batch_size=2,  # specify your max batch size
        max_workspace_size_bytes=2 * (10 ** 9),  # specify the max workspace
        precision_mode="FP32")  # precision, can be "FP32" (32 floating point precision) or "FP16"
    
    # write the TensorRT model to be used later for inference
    with gfile.FastGFile(os.path.join(save_dir, "TensorRT_model.pb"), 'wb') as f:
        f.write(trt_graph.SerializeToString())
    print("TensorRT model is successfully stored!")
    # check how many ops of the original frozen model
    all_nodes_frozen_graph = len([1 for n in frozen_graph.node])
    # check how many ops that is converted to TensorRT engine
    trt_engine_nodes = len([1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
    print("numb. of trt_engine_nodes in TensorRT graph:", trt_engine_nodes)
    all_nodes = len([1 for n in trt_graph.node])
    
    print("numb. of all_nodes in frozen graph:", all_nodes_frozen_graph)
    print("numb. of all_nodes in TensorRT graph:", all_nodes)


def read_pb_graph(model):
    with gfile.FastGFile(model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


def inference_from_tensor_rt_graph(log_dir, dataset_test):
    with tf.variable_scope('test_iterator'):
        iterator_data_test = dataset_test.make_initializable_iterator()
        next_test_data = iterator_data_test.get_next()
    
    # Set up model
    initializers = [i.initializer for i in
                    [iterator_data_test]]
    
    save_dir = os.path.join(log_dir, 'pretrained')
    TENSORRT_MODEL_PATH = os.path.join(save_dir, 'TensorRT_model.pb')
    
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.50))) as sess:
            for init in initializers:
                sess.run(init)
            sess.run(tf.global_variables_initializer())
            # read TensorRT model
            trt_graph = read_pb_graph(TENSORRT_MODEL_PATH)
            
            # obtain the corresponding input-output tensor
            tf.import_graph_def(trt_graph, name='')
            input = sess.graph.get_tensor_by_name("Sequential_model_1/inference/strided_slice:0")
            output = sess.graph.get_tensor_by_name("Sequential_model_1/inference/Sigmoid:0")
            
            time_list = []
            for i in range(10):
                start_time = time.time()
                outputs = sess.run([output])
                elapsed_time = time.time() - start_time
                if i > 0:
                    time_list.append(elapsed_time)
            print(f'average time taken for inference per data is  {np.mean(time_list)}')
            
            # in this case, it demonstrates to perform inference for 50 times
            total_time = 0
            n_time_inference = 50
            out_pred = sess.run(output, feed_dict={input: next_test_data})
            for i in range(n_time_inference):
                t1 = time.time()
                out_pred = sess.run(output, feed_dict={input: next_test_data})
                t2 = time.time()
                delta_time = t2 - t1
                total_time += delta_time
                print("needed time in inference-" + str(i) + ": ", delta_time)
            avg_time_tensorRT = total_time / n_time_inference
            print("average inference time: ", avg_time_tensorRT)
    
    # variable
    FROZEN_MODEL_PATH = os.path.join(save_dir, 'frozen_model.pb')
    
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            # read TensorRT model
            frozen_graph = read_pb_graph(FROZEN_MODEL_PATH)
            
            # obtain the corresponding input-output tensor
            tf.import_graph_def(frozen_graph, name='')
            input = sess.graph.get_tensor_by_name("Sequential_model_1/inference/strided_slice:0")
            output = sess.graph.get_tensor_by_name("Sequential_model_1/inference/Sigmoid:0")
            
            # in this case, it demonstrates to perform inference for 50 times
            total_time = 0
            n_time_inference = 50
            out_pred = sess.run(output, feed_dict={input: next_test_data})
            for i in range(n_time_inference):
                t1 = time.time()
                out_pred = sess.run(output, feed_dict={input: next_test_data})
                t2 = time.time()
                delta_time = t2 - t1
                total_time += delta_time
                print("needed time in inference-" + str(i) + ": ", delta_time)
            avg_time_original_model = total_time / n_time_inference
            print("average inference time: ", avg_time_original_model)
            print("TensorRT improvement compared to the original model:", avg_time_original_model / avg_time_tensorRT)
