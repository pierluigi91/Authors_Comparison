#! /usr/bin/env python

import sys
sys.path.append('pre_processing')
import pre_processing.clean_text as clean_text
import json
from operator import itemgetter

import numpy as np
import tensorflow as tf
sys.path.append('deep_learning')
from deep_learning import data_helpers
from deep_learning.text_cnn import TextCNN
sys.path.append('')
from machine_learning.naive_bayes.eval import evaluation as nb_ev
from machine_learning.spark import spark_text as sp

precision = []
recall = []
f1_score = []
pos = []

num_authors = 13

autori = None
length_opere = []
with open('./data/authors.json') as data_file:
    js = json.load(data_file)
for d in js:
    opera = open('./data/stemmed_data/'+d['file_name'], "r").readlines()
    print "LUNGHEZZA DI " + str(d['file_name']) + " = " + str(len(opera))
    #CAMBIARE QUI PER LE OPERE
    length_opere.append(len(opera))
max_length = max(length_opere)


# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 1, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS.batch_size
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
padding_word = '<PAD/>'
# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
x, y, vocabulary, vocabulary_inv = data_helpers.load_data()
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
# Split train/test set
# TODO: This is very crude, should use cross-validation
x_train, x_dev = x_shuffled[:-1000], x_shuffled[-1000:]
y_train, y_dev = y_shuffled[:-1000], y_shuffled[-1000:]
print("Vocabulary Size: {:d}".format(len(vocabulary)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


def print_results(array):
    with open('./data/authors.json') as data_file:
        data = json.load(data_file)
    array = array.tolist()
    results_list = []
    sum = 0.0

    for idx, val in enumerate(array):

        temp_len = len(open('data/stemmed_data/'+data[idx]['file_name'], "r").readlines())
        norm = (val * max_length) / temp_len
        results_list.append([data[idx]['name'], data[idx]['surname'], norm])
        if val > 0.0:
            sum += norm


    results_list = sorted(results_list, key=itemgetter(2), reverse=True)
    print('=============================================')
    print('Nome\t|\tCognome\t\t|\tScore')
    print()
    for result in results_list:
        #print(result[0], '\t', result[1], '\t', result[2])
        if result[2] > 0.0:
            print("%-14s %-8s %20.4f" % (result[0], result[1], (result[2]/sum)*100.0)+" %")
    print('=============================================')


def evaluation(path):
    idx = 1
    sentences = list(open(path, "r").readlines())
    sentences = [s.strip() for s in sentences]

    sentences = [data_helpers.clean_str(sent) for sent in sentences]
    sentences = [clean_text.stemming(sent) for sent in sentences]
    sentences = [s.split(" ") for s in sentences]

    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)

    label = y[idx]
    x_batch = np.array([[vocabulary[word] if word in vocabulary else vocabulary['<PAD/>'] for word in padded_sentence] for padded_sentence in padded_sentences])
    a = pred(x_batch, label, sequence_length, multiple_lines=True)
    print "ARRAY DI EVALUATION: " + str(a)
    a = a / a.max(axis=0)
    print_results(a)


# Training
# ==================================================
def pred(entrada, label, seq_len, multiple_lines=False):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                # sequence_length=x_train.shape[1],
                sequence_length=seq_len,
                num_classes=num_authors,
                vocab_size=len(vocabulary),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            saver = tf.train.Saver(tf.all_variables())

            # Initialize all variables
            sess.run(tf.initialize_all_variables())
            saver.restore(sess, "runs/1468083035/checkpoints/model-00")

            def predict_step(x_batch):
                """
                predict on a test set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                y_pred = sess.run(
                    [cnn.predictions, cnn.scores], feed_dict)
                return (y_pred)

            if not multiple_lines:
                entrada = np.reshape(entrada, (-1, len(entrada)))
                y = predict_step(entrada)
                return y[1][0]
            else:
                all_values = np.zeros(num_authors)
                label = np.reshape(label, (-1, len(label)))
                for line in entrada:
                    line = np.reshape(line, (-1, len(line)))
                    all_values = np.vstack([all_values, predict_step(line)[1][0]])
                all_values = np.delete(all_values, 0, axis=0)
                return np.mean(all_values, axis=0)


from threading import Thread


def start():
    path = raw_input("Inserire un path di un file da classificare: ")
    #Thread(target=evaluation(path))
    #Thread(target=sp.evaluate(path))
    #Thread(target=nb_ev.eval(path))
    #sp_res = np.array(sp.evaluate(path))
    #nb_res = np.array(nb_ev.eval(path))
    #clean_text.parsing(path)
    #tf_res = np.array(evaluation(path))

    print 'SPARK: '
    #print sp_res
    print ''
    print 'BAYES: '
    #print nb_res
    print ''
    print 'TENSORFLOW: '
    np.array(evaluation(path))
    print ''
    print 'RISULTATO:'
    # sp_nb_mean = np.add(sp_res*0.5, nb_res*0.5)
    # result = sp_nb_mean * tf_res
    # print result




def _start_shell(local_ns=None):
  import IPython
  user_ns = {}
  if local_ns:
    user_ns.update(local_ns)
  user_ns.update(globals())
  IPython.start_ipython(argv=[], user_ns=user_ns)


_start_shell(locals())