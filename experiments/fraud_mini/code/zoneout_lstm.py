import tensorflow as tf


class ZoneoutLSTMCell(tf.nn.rnn_cell.RNNCell):
    '''Wrapper for tf LSTM to create Zoneout LSTM Cell

    inspired by:
    https://github.com/teganmaharaj/zoneout/blob/master/zoneout_tensorflow.py

    Published by one of 'https://arxiv.org/pdf/1606.01305.pdf' paper writers.

    Many thanks to @Ondal90 for pointing this out. You sir are a hero!
    '''
    
    def __init__(self, num_units, is_training, zoneout_factor_cell=0., zoneout_factor_output=0., state_is_tuple=True, name=None):
        '''Initializer with possibility to set different zoneout values for cell/hidden states.
        '''
        zm = min(zoneout_factor_output, zoneout_factor_cell)
        zs = max(zoneout_factor_output, zoneout_factor_cell)
        
        if zm < 0. or zs > 1.:
            raise ValueError('One/both provided Zoneout factors are not in [0, 1]')
        
        self._cell = tf.nn.rnn_cell.LSTMCell(num_units, forget_bias=-2., state_is_tuple=state_is_tuple, name=name)
        self._zoneout_cell = zoneout_factor_cell
        self._zoneout_outputs = zoneout_factor_output
        self.is_training = is_training
        self.state_is_tuple = state_is_tuple
    
    @property
    def state_size(self):
        return self._cell.state_size
    
    @property
    def output_size(self):
        return self._cell.output_size
    
    def __call__(self, inputs, state, scope=None):
        '''Runs vanilla LSTM Cell and applies zoneout.
        '''
        # Apply vanilla LSTM
        output, new_state = self._cell(inputs, state, scope)
        
        if self.state_is_tuple:
            (prev_c, prev_h) = state
            (new_c, new_h) = new_state
        else:
            num_proj = self._cell._num_units if self._cell._num_proj is None else self._cell._num_proj
            prev_c = tf.slice(state, [0, 0], [-1, self._cell._num_units])
            prev_h = tf.slice(state, [0, self._cell._num_units], [-1, num_proj])
            new_c = tf.slice(new_state, [0, 0], [-1, self._cell._num_units])
            new_h = tf.slice(new_state, [0, self._cell._num_units], [-1, num_proj])
        
        # Apply zoneout
        if self.is_training:
            # nn.dropout takes keep_prob (probability to keep activations) not drop_prob (probability to mask activations)!
            c = (1 - self._zoneout_cell) * tf.nn.dropout(new_c - prev_c, (1 - self._zoneout_cell)) + prev_c
            h = (1 - self._zoneout_outputs) * tf.nn.dropout(new_h - prev_h, (1 - self._zoneout_outputs)) + prev_h
        
        else:
            c = (1 - self._zoneout_cell) * new_c + self._zoneout_cell * prev_c
            h = (1 - self._zoneout_outputs) * new_h + self._zoneout_outputs * prev_h
        
        new_state = tf.nn.rnn_cell.LSTMStateTuple(c, h) if self.state_is_tuple else tf.concat(1, [c, h])
        
        return output, new_state
