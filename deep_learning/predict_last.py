#! /usr/bin/env python

import json
import sys
from operator import itemgetter

import numpy as np
import tensorflow as tf

import data_helpers
from text_cnn import TextCNN

sys.path.append('../pre_processing')

from pre_processing import clean_text

precision = []
recall = []
f1_score = []
pos=[]

num_authors = 13

js = None
autori= None
length_opere=[]
with open('data/authors.json') as data_file:
    js = json.load(data_file)
for d in js:
    opera = open('data/input_stemmed/'+d['file_name'], "r").readlines()  #CAMBIARE QUI PER LE OPERE
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
    sum=0.0

    for idx, val in enumerate(array):
        temp_len = len(open('data/input_stemmed/'+d['file_name'], "r").readlines())
        norm = (val * max_length) / temp_len
        results_list.append([data[idx]['name'], data[idx]['surname'], norm])
        if val>0.0:
            sum+=norm


    results_list = sorted(results_list, key=itemgetter(2), reverse=True)
    print('=============================================')
    print('Nome\t|\tCognome\t\t|\tScore')
    print()
    for result in results_list:
        #print(result[0], '\t', result[1], '\t', result[2])
        if result[2]>0.0:
            print("%-14s %-8s %20.4f" % (result[0], result[1], (result[2]/sum)*100.0)+" %")
    print('=============================================')
    # print "PRECISION", sum(precision)/len(precision)
    # print "RECALL", sum(recall)/len(recall)
    # print "F1_MEASURE", (sum(precision)/len(precision) * sum(recall)/len(recall)) / (sum(precision)/len(precision) + sum(recall)/len(recall))




def sentence_fenno(sent, in_file=False):
    idx = 1
    entrada = x[idx]
    sentence = data_helpers.clean_str(sent).split()
    seq_len = max(5, len(sentence))
    for i in range(seq_len - len(sentence)):
        sentence.append('<PAD/>')

    entrada = np.array([vocabulary[w] for w in sentence if w in vocabulary])

    empty_check = False
    for w in sentence:
        if w in vocabulary:
            if w != padding_word:
                empty_check = True
        else:
            seq_len -= 1

    #empty_check = [i += 1 for w in sentence if w in vocabulary and w != '<PAD/>']
    if not empty_check:
        print('Nessuna parola presente nel vocabolario')
        return
    label = y[idx]
    # print(entrada)
    # print(' '.join([vocabulary_inv[i] for i in entrada]))
    # print('sequence length', seq_len)
    #pred(entrada, label, seq_len)
    a = pred(entrada, label, seq_len)
    if not in_file:
        print_results(a)
        a = a / a.max(axis=0)
    else:
        return a


def file_fenno_2(path):
    idx = 1
    sentences = list(open(path, "r").readlines())
    sentences = [s.strip() for s in sentences]

    sentences = [data_helpers.clean_str(sent) for sent in sentences]
    sentences = [s.split(" ") for s in sentences]

    sentences = [[clean_text.stemming(word) for _ in sentences] for word in s]

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
            #saver.restore(sess, "/home/pierluigi/PycharmProjects/Authors_Comparison/deep_learning/runs/1464774069/checkpoints/model-5400")
            #saver.restore(sess, "/home/pierluigi/PycharmProjects/Authors_Comparison/deep_learning/runs/1464950406/checkpoints/model-3900")
            saver.restore(sess, "runs/1464964595/checkpoints/model-3900")


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

                # y_true = np.reshape(label, (-1, len(label)))
                #
                # approx=[]
                # for p in y_pred[1][0]:
                #     approx.append(p/np.amax(y_pred[1][0]))
                #
                # approx_ult=[]
                # for p in approx:
                #     if p!=1.0:
                #         approx_ult.append(0.0)
                #     else:
                #         approx_ult.append(p)
                #
                # print y_true
                # print approx_ult
                #
                # precision.append(sk.metrics.precision_score(y_true, approx_ult,average='binary'))
                # recall.append(sk.metrics.recall_score(y_true, approx_ult,average='binary'))
                #
                # if (sk.metrics.precision_score(y_true, approx_ult,average='binary') ==1.0):
                #     pos.append(1)
                #f1_score.append(sk.metrics.f1_score(y_true, approx_ult,average='binary'))

                return (y_pred)

            if not multiple_lines:
                entrada = np.reshape(entrada, (-1, len(entrada)))
                label = np.reshape(label, (-1, len(label)))
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


            #print (y[1][0])    file_fenno_2('data/rt-polaritydata/rt-polarity.test')
            #return(y[1][0] / y[1][0].max(axis=0))
            #return y[1][0]

def _start_shell(local_ns=None):
  # An interactive shell is useful for debugging/development.
  import IPython
  user_ns = {}
  if local_ns:
    user_ns.update(local_ns)
  user_ns.update(globals())
  IPython.start_ipython(argv=[], user_ns=user_ns)

#file_fenno_2("/home/pierluigi/Scrivania/testi/prove/hp_01.txt")

_start_shell(locals())