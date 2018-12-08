import tensorflow as tf
from model import BILSTM_CRF
import time
import os
import datetime
import pandas as pd
import preprocessing
from preprocessing import tag2label
from tensorflow.contrib.crf import viterbi_decode
import sklearn
import numpy as np
import data_helpers

tf.flags.DEFINE_bool("allow_soft_placement", True, "allow_soft_placement")
tf.flags.DEFINE_bool("log_device_placement", False, "log_device_placement")
tf.flags.DEFINE_integer("batch_szie", 64, "batch_size")
tf.flags.DEFINE_integer("num_checkpoints", 5, "number of checkpoints to store(default : 5)")
tf.flags.DEFINE_float("dropout", 0.5, "dropout value")
tf.flags.DEFINE_integer("num_epochs", 200, "default num of epoch")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "dropout property")
tf.flags.DEFINE_float("evaluate_every", 1000, "evaluate for every 1000 steps")
tf.flags.DEFINE_float("checkpoint_every", 1000, "save check point for every 1000 steps")
FLAGS = tf.flags.FLAGS


def read_data(file_name):
    data = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            data.append(line)
        f.close()
    return data


def get_batch_length(x):
    length = []
    for x_ in x:
        length.append(len(x_))
    return length

def train(x_train, y_train, x_dev, y_dev, embedding, tag2label, length_train, length_test):
    # Training
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5)
        session_config = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement,
            gpu_options=gpu_options
        )
        sess = tf.Session(config=session_config)
        with sess.as_default():
            model = BILSTM_CRF(batch_size=FLAGS.batch_size,
                               hidden_dim=100,
                               embeddings=embedding,
                               CRF=True,
                               dropout=FLAGS.dropout,
                               num_tags=len(tag2label))
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradoent values and sparsity
            grad_summaries = []
            for g,v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)

            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summaries = tf.summary.scalar("loss", model.loss)

            # Train Summaries
            train_summary_op = tf.summary_merge([loss_summaries, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev Summaries
            dev_summary_op = tf.summary.merge([loss_summaries])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory.
            checkpoint_dir = os.path.abspath(os.path.join(out_dir), "checkpoints")
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch, length):
                """

                :param x_batch:
                :param y_batch:
                :param length:
                :return:
                """
                feed_dict = {
                    model.word_ids : x_batch,
                    model.labels : y_batch,
                    model.sequence_length : length
                }
                _, step, summaries, loss, logits, transition_params = sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.logits, model.transition_params],
                    feed_dict
                )

                time_str = datetime.datetime.now().isoformat()
                if step % 100 == 0:
                    label_list = []
                    accs = []
                    f1_scores = []
                    for logit, seq_len in zip(logits, length):
                        viterbi_seq, _ = viterbi_decode(logits[:seq_len], transition_params)
                        label_list.append(viterbi_seq)
                    for lab, label_pred, length_sub in zip(y_batch, label_list, length):
                        lab = lab[:length_sub]
                        label_pred = label_pred[:length_sub]
                        accs += [a==b for (a,b) in zip(lab, label_pred)]
                        f1_score = sklearn.metrics.f1_score(lab, label_pred, average="micro")
                        f1_scores += [f1_score]

                    acc = np.mean(accs)
                    f_score = np.mean(f1_scores)
                    print("{} : step {} loss {:g}, acc {:g}, f_score {:g}".format(time_str, step, loss, 100 * acc, 100 * f1_score))

                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, length, writer = None):
                """

                :param x_batch:
                :param y_batch:
                :param length:
                :param writer:
                :return:
                """
                feed_dict = {
                    model.word_ids: x_batch,
                    model.labels: y_batch,
                    model.sequence_length: length
                }
                step, summaries, loss, logits, transition_params = sess.run(
                    [global_step, dev_summary_op, model.loss, model.logits, model.transition_params],
                    feed_dict
                )
                time_str = datetime.datetime.now().isoformat()

                label_list = []
                accs = []
                f1_scores = []
                for logit, seq_len in zip(logits, length):
                    viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                    label_list.append(viterbi_seq)
                for lab, label_pred, length_sub in zip(y_batch, label_list, length):
                    lab = lab[:length_sub]
                    label_pred = label_pred[:length_sub]
                    accs += [a==b for (a,b) in zip(lab, label_pred)]
                    f1_score = sklearn.metrics.f1_score(lab, label_pred, average="micro")
                    f1_scores += [f1_score]

                acc = np.mean(accs)
                f_score = np.mean(f1_scores)
                print("test {}: step {} loss {:g}, acc {:g}, f_score {:g}".format(time_str, step, loss, 100 * acc, 100 * f_score))
                if writer:
                    writer.add_summary(summaries, step)
            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train, length_train)), FLAGS.batch_size, FLAGS.num_epochs
            )

            # Training loop, For each batch..
            for batch in batches:
                x_batch, y_batch, length_batch = zip(*batch)
                length = length_batch
                train_step(x_batch, y_batch, length_batch)
                current_step = tf.train.global_step(sess, global_step)

                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    test_batches = data_helpers.batch_iter(list(zip(x_dev, y_dev, length_test)), FLAGS.batch_size, 1)
                    for test_batch in test_batches:
                        x_test_batch, y_test_batch, length_test_batch = zip(*test_batch)
                        dev_step(x_test_batch, y_test_batch, length_test_batch, writer=dev_summary_writer)

                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("save model checkpoint to {}\n".format(path))


def main(argv=None):
    vocab = pd.read_pickle('./data/vocab.pkl')
    sentences = read_data('./data/train_sentence.txt')
    labels = read_data('./data/train_label.txt')
    sentence_id, label_id, real_length = preprocessing.generate_mapping(sentences, labels, vocab)
    train_content, train_label, test_content, test_label, train_length, test_length, valid_length, valid_content, valid_label = data_helpers.get_train_test_data(sentence_id, label_id, real_length)
    word2id = vocab[0]
    embedding_matrix = preprocessing.generate_embeddings(word2id, vocab[0], './data/embedding/vec.txt', embedding_dim=100)
    train(train_content, train_label, valid_content, valid_label,embedding_matrix, tag2label, train_length, valid_length)


if __name__ == "__main__":
    tf.app.run()
