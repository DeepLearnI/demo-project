"""
Copyright (C) DeepLearning Financial Technologies Inc. - All Rights Reserved
Unauthorized copying, distribution, reproduction, publication, use of this file, via any medium is strictly prohibited.
Proprietary and confidential – June 2019
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf

import os
import time

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_curve

plt.switch_backend('agg')


def evaluate_capture_rate_tensorflow(targets, predictions, threshold=0.0001, cut_ratio=0.002):
    weights = (1 - predictions) * 39 + 1
    
    total_num_cases = tf.reduce_sum(weights)
    cutoff = cut_ratio * total_num_cases
    
    indices = tf.argsort(predictions, axis=0, direction='DESCENDING')
    targets = tf.gather(targets, indices)
    predictions = tf.gather(predictions, indices)
    
    counter = tf.where(tf.equal(targets, 1),
                       tf.ones_like(targets),
                       40 * tf.ones_like(targets))
    
    stop_index = tf.where(tf.greater_equal(tf.cumsum(counter), cutoff))[0][0]
    sub_predictions = predictions[:stop_index]
    sub_targets = targets[:stop_index]
    
    captured = tf.where(tf.equal(sub_targets, 1),
                        tf.where(tf.greater_equal(sub_predictions, threshold),
                                 tf.ones_like(sub_targets),
                                 tf.zeros_like(sub_targets)),
                        tf.zeros_like(sub_targets))
    
    capture_rate = tf.reduce_sum(captured) / tf.reduce_sum(targets)
    return capture_rate, stop_index


def evaluate_capture_rate_for_sampled(df, ca, cs, cw, threshold=0.0000000001, weight=40, cut_ratio=0.002):
    total_num_cases = df[cw].sum()  # assume that df[cw] has only 1, or weight as values!! Otherwise it is incorrect
    cutoff = cut_ratio * total_num_cases
    df = df.sort_values([cs], ascending=False)
    
    # return df[ca].iloc[:int(cutoff)].sum() / df[ca].sum()
    
    answers = np.array(df[ca])
    scores = np.array(df[cs])
    weights = np.array(df[cw])
    counter = 0
    captured = 0
    index_pointer = 0
    
    while counter < cutoff:
        counter += weights[index_pointer]
        if answers[index_pointer] == 1:
            if scores[index_pointer] >= threshold:
                captured += 1
        
        index_pointer += 1
    try:
        capture_rate = captured / answers[answers == 1].shape[0]
    except:
        capture_rate = np.nan
    return capture_rate


def evaluate_and_plot(y_pred, y_test, index, model_name='xgb_with_sampled_dataset', threshold=0.5, weight=40, out_dir='', step=None, use_tf=True, sess=None):
    start = time.time()
    y_test = np.round(y_test)
    if use_tf:
        assert sess is not None
        y_test_placeholder = tf.placeholder(tf.float32, shape=(None,), name='capture_rate_labels')
        y_pred_placeholder = tf.placeholder(tf.float32, shape=(None,), name='capture_rate_targets')
        
        capture_rate_df, stop_index_df = evaluate_capture_rate_tensorflow(y_test_placeholder, y_pred_placeholder, weight=weight)
        
        capture_rate, stop_index = sess.run([capture_rate_df, stop_index_df], feed_dict={y_test_placeholder: y_test, y_pred_placeholder: y_pred})
    
    else:
        pred = pd.DataFrame({'pred': y_pred,
                             'label': y_test,
                             'weight': (1 - y_test) * (weight - 1) + 1},
                            index=index)
        
        capture_rate = evaluate_capture_rate_for_sampled(pred, 'label', 'pred', 'weight',
                                                         weight=weight)
        
        pred.to_csv('scores.csv')
    
    end = time.time()
    print('Capture rate computation time: {}sec'.format(end - start))
    print('CAPTURE RATE:\n{}'.format(capture_rate))
    
    y_score = y_pred[:]
    
    # convert into binary values
    y_pred = np.where(y_pred >= threshold, 1, 0)
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Accuracy
    accuracy = accuracy_score(y_pred, y_test)
    print('Confusion matrix:\n {}'.format(cm))
    print('Accuracy:\n {}'.format(accuracy))
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    
    from sklearn.metrics import average_precision_score
    average_precision = average_precision_score(y_test, y_score)
    
    print('Average precision-recall score: {0:0.2f}'.format(
        average_precision))
    plt.step(recall, precision, color='b', alpha=0.3,
             where='post')
    plt.fill_between(recall, precision, alpha=0.3, color='b')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))
    
    fig_name = (os.path.join(out_dir, 'pr_curve_for_{}.png'.format(model_name)) if step is None else
                os.path.join(out_dir, 'pr_curve_for_{}_step_{}.png'.format(model_name, step)))
    plt.savefig(fig_name)
    
    print('CAPTURE RATE:\n{}'.format(capture_rate))
    return capture_rate * 100, fig_name
