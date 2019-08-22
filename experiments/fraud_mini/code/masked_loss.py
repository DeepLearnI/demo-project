"""
Copyright (C) DeepLearning Financial Technologies Inc. - All Rights Reserved
Unauthorized copying, distribution, reproduction, publication, use of this file, via any medium is strictly prohibited.
Proprietary and confidential – June 2019
"""


import tensorflow as tf



def sequence_mask(lengths, max_len, expand=True):
	'''Returns a 2-D or 3-D tensorflow sequence mask depending on the argument 'expand'
	'''
	if expand:
		return tf.expand_dims(tf.sequence_mask(lengths, maxlen=max_len, dtype=tf.float32), axis=-1)
	return tf.sequence_mask(lengths, maxlen=max_len, dtype=tf.float32)

def MaskedSigmoidCrossEntropy(targets, outputs, targets_lengths, hparams, mask=None):
	'''Computes a masked SigmoidCrossEntropy with logits
	'''
	#target shape: [batch_size, time, 1]
	#mask with expansion: [batch_size, time, 1]

	#[batch_size, time_dimension]
	#example:
	#sequence_mask([1, 3, 2], False) = [[1., 0., 0.],
	#							        [1., 1., 1.],
	#							        [1., 1., 0.]]
	#Will be used to mask loss values
	if mask is None:
		mask = sequence_mask(targets_lengths, max_len=tf.shape(targets)[1], expand=True)

	with tf.control_dependencies([tf.assert_equal(tf.shape(targets), tf.shape(mask))]):
		#Use a weighted sigmoid cross entropy to measure the <stop_token> loss. Set hparams.pos_weight to 1
		#will have the same effect as  vanilla tf.nn.sigmoid_cross_entropy_with_logits.
		losses = tf.nn.weighted_cross_entropy_with_logits(targets=targets, logits=outputs, pos_weight=hparams.pos_weight)

	with tf.control_dependencies([tf.assert_equal(tf.shape(mask), tf.shape(losses))]):
		masked_loss = losses * mask #element wise

	return tf.reduce_sum(masked_loss) / tf.cast(tf.reduce_sum(mask), tf.float32)

def MaskedFPPenalty(targets, outputs, targets_lengths, hparams, mask=None):
	'''Computes a masked penlize false positives with outputs (after sigmoid)
	'''
	#target shape: [batch_size, time, 1]
	#mask with expansion: [batch_size, time, 1]

	#[batch_size, time_dimension]
	#example:
	#sequence_mask([1, 3, 2], False) = [[1., 0., 0.],
	#							        [1., 1., 1.],
	#							        [1., 1., 0.]]
	#Will be used to mask loss values
	if mask is None:
		mask = sequence_mask(targets_lengths, max_len=tf.shape(targets)[1], expand=True)

	with tf.control_dependencies([tf.assert_equal(tf.shape(targets), tf.shape(mask))]):
		if hparams.flip_targets:
			losses = targets * tf.exp(1. - outputs)

		else:
			losses = (1. - targets) * tf.exp(outputs)
			
	with tf.control_dependencies([tf.assert_equal(tf.shape(mask), tf.shape(losses))]):
		masked_loss = losses * mask #element wise

	return hparams.penalty_weight * (tf.reduce_sum(masked_loss) / tf.cast(tf.reduce_sum(mask), tf.float32))