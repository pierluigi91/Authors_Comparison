#! /usr/bin/env python

import sys
sys.path.append('pre_processing')
import pre_processing.clean_text as clean_text
import json
from operator import itemgetter
import re
import numpy as np
import tensorflow as tf
sys.path.append('deep_learning')
from deep_learning import data_helpers
from deep_learning.text_cnn import TextCNN
sys.path.append('')
from machine_learning.naive_bayes.eval import evaluation as nb_ev
from machine_learning.spark import spark_text as sp
import math


num_authors = 13

autori = None
length_opere = []
with open('./data/authors.json') as data_file:
    js = json.load(data_file)
for d in js:
    opera = open('./data/stemmed_data/'+d['file_name'], "r").readlines()
    print "LUNGHEZZA DI " + str(d['file_name']) + " = " + str(len(opera))
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

    result_array = []
    for idx, val in enumerate(array):

        norm = val
        results_list.append([data[idx]['name'], data[idx]['surname'], norm])
        result_array.append(norm)
        if val >= 0.0:
            sum += norm

    results_list = sorted(results_list, key=itemgetter(2), reverse=True)
    print('RISULTATO')
    print('=============================================')
    print('Nome\t|\tCognome\t\t|\tScore')
    for result in results_list:
        if result[2] > 0.0:
            print("%-14s %-8s %20.4f" % (result[0], result[1], (result[2]/sum)*100.0)+" %")
            print('=============================================')


def evaluation(path):
    idx = 1
    sentences = list(open(path, "r").readlines())
    sentences = [s.strip() for s in sentences]
    sentences = [data_helpers.clean_str(sent) for sent in sentences]
    sentences = [clean_text.stemming(sent) for sent in sentences]
    sentences = [re.sub('[^0-9a-zA-Z\s]+', '', s) for s in sentences]

    def lunghezza_frase(s):
        lung_temp = 0
        for w in s.split():
            lung_temp += 1
        return lung_temp

    sequence_length = 0
    for s in sentences:
        lung_frase = lunghezza_frase(s)
        if lung_frase > sequence_length:
            sequence_length = lung_frase

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        lung = lunghezza_frase(sentence)
        num_padding = sequence_length - lung
        new_sentence = []
        for word in sentence.split():
            new_sentence.append(str(word))
        for j in range(0, num_padding):
            new_sentence.append(padding_word)
        padded_sentences.append(new_sentence)

    label = y[idx]
    x_batch = np.array([[vocabulary[word] if word in vocabulary else vocabulary['<PAD/>'] for word in padded_sentence] for padded_sentence in padded_sentences])
    a = pred(x_batch, label, sequence_length, multiple_lines=True)
    print "ARRAY DI EVALUATION: " + str(a)
    a = a + abs(a.min(axis=0))
    return a


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
            saver.restore(sess, "runs/1468112105/checkpoints/model-82100")

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

# Metodo usato per debuggare, gli dai in input x libri da valutare. Va modificato ad hoc lo start pero!
def start2():
    start("/Users/Max/PycharmProjects/Authors_Comparison/data/sentence_to_line_data/Howard_Phillips_Lovecraft/The_Case_of_Charles_Dexter_W.txt", "LOVECRAFT")
    start("/Users/Max/PycharmProjects/Authors_Comparison/data/sentence_to_line_data/Robert_Louis_Stevenson/The_Body-Snatcher.txt", "STEVENSON")
    start("/Users/Max/PycharmProjects/Authors_Comparison/data/sentence_to_line_data/Charles_Dickens/The_Battle_of_Life.txt", "DICKENS")
    start("/Users/Max/PycharmProjects/Authors_Comparison/data/sentence_to_line_data/James_Joyce/Chamber_Music.txt", "JOYCE")
    start("/Users/Max/PycharmProjects/Authors_Comparison/data/sentence_to_line_data/Mark_Twain/Alonzo_Fitz.txt", "MARK TWAIN")
    start("/Users/Max/PycharmProjects/Authors_Comparison/data/sentence_to_line_data/Jonathan_Swift/A_Modest_Proposal.txt", "SWIFT")
    start("/Users/Max/PycharmProjects/Authors_Comparison/data/sentence_to_line_data/Mark_Twain/mark_twain.txt", "MARK TWAIN")
    start("/Users/Max/PycharmProjects/Authors_Comparison/data/sentence_to_line_data/Charles_Dickens/acc.txt", "DICKENS")
    start("/Users/Max/PycharmProjects/Authors_Comparison/data/sentence_to_line_data/Charles_Dickens/oliver_twist.txt", "DICKENS")
    start("/Users/Max/PycharmProjects/Authors_Comparison/data/sentence_to_line_data/Jane_Austen/jane_austen.txt", "AUSTEN")


def start():
    path = raw_input("Inserire un path di un file da classificare: ")
    #Thread(target=evaluation(path))
    #Thread(target=sp.evaluate(path))
    #Thread(target=nb_ev.eval(path))
    sp_res = np.array(sp.evaluate(path))
    nb_res = np.array(nb_ev.eval(path))
    tf_res = np.array(evaluation(path))
    tf_res2 = []
    for el in tf_res:
        tf_res2.append((el - min(tf_res))/(max(tf_res - min(tf_res))))
    tf_res2 = np.array(tf_res2)
    tf_res3 = []
    with open('./data/authors.json') as data_file:
        data = json.load(data_file)
    for idx, val in enumerate(tf_res2):
        temp_len = len(open('data/stemmed_data/'+data[idx]['file_name'], "r").readlines())
        norm = (val * math.log(max_length)) / math.log(temp_len)
        tf_res3.append(norm)
    tf_res3 = np.array(tf_res3)
    print 'SPARK: '
    print sp_res
    print ''
    print 'BAYES: '
    print nb_res
    print ''
    print 'TENSORFLOW: '
    print tf_res
    print ''
    print 'TENSORFLOW NORMALIZZATO: '
    print tf_res3
    print ''
    print 'RISULTATO : '
    sp_nb_mean = np.add(sp_res*0.5, nb_res*0.5)
    result = sp_nb_mean * tf_res3
    print_results(result)



def _start_shell(local_ns=None):
  import IPython
  user_ns = {}
  if local_ns:
    user_ns.update(local_ns)
  user_ns.update(globals())
  IPython.start_ipython(argv=[], user_ns=user_ns)


_start_shell(locals())