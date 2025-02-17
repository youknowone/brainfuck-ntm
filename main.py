from __future__ import absolute_import

import importlib
import tensorflow as tf
from ntm_cell import NTMCell
from ntm import NTM
from utils import pp

flags = tf.app.flags
flags.DEFINE_string("task", "copy", "Task to run [copy, recall]")
flags.DEFINE_integer("epoch", 100000, "Epoch to train [100000]")
flags.DEFINE_integer("input_dim", 10, "Dimension of input [10]")
flags.DEFINE_integer("output_dim", 10, "Dimension of output [10]")
flags.DEFINE_integer("min_length", 1, "Minimum length of input sequence [1]")
flags.DEFINE_integer("max_length", 10, "Maximum length of output sequence [10]")
flags.DEFINE_integer("controller_layer_size", 1, "The size of LSTM controller [1]")
flags.DEFINE_integer("controller_dim", 100, "The dimension of LSTM controller [100]")
flags.DEFINE_integer("write_head_size", 1, "The number of write head [1]")
flags.DEFINE_integer("read_head_size", 1, "The number of read head [1]")
flags.DEFINE_integer("test_max_length", 120, "Maximum length of output sequence [120]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("continue_train", None, "True to continue training from saved checkpoint. False for restarting. None for automatic [False]")
FLAGS = flags.FLAGS


def create_ntm(FLAGS, sess, **ntm_args):
    cell = NTMCell(
        input_dim=FLAGS.input_dim,
        output_dim=FLAGS.output_dim,
        controller_layer_size=FLAGS.controller_layer_size,
        controller_dim=FLAGS.controller_dim,
        write_head_size=FLAGS.write_head_size,
        read_head_size=FLAGS.read_head_size)
    scope = ntm_args.pop('scope', 'NTM-%s' % FLAGS.task)
    ntm = NTM(
        cell, sess, FLAGS.min_length, FLAGS.max_length,
        test_max_length=FLAGS.test_max_length, scope=scope, **ntm_args)
    return cell, ntm


def main(_):
    with tf.device('/cpu:0'), tf.Session() as sess:
        try:
            task = importlib.import_module('tasks.%s' % FLAGS.task)
        except ImportError:
            print("task '%s' does not have implementation" % FLAGS.task)
            raise

        if hasattr(task, 'preset_flags'):
            task.preset_flags(FLAGS)

        pp.pprint(flags.FLAGS.__flags)

        if FLAGS.is_train:
            cell, ntm = create_ntm(FLAGS, sess)
            task.train(ntm, FLAGS, sess)
        else:
            cell, ntm = create_ntm(FLAGS, sess, forward_only=True)

        # if FLAGS.task.startswith('brainfuck'):
        #     FLAGS.checkpoint_name = 'brainfuck'

        ntm.load(FLAGS.checkpoint_dir, FLAGS.task)

        if FLAGS.task == 'copy':
            #task.run(ntm, FLAGS.test_max_length * 1 / 3, sess)
            #print
            #task.run(ntm, FLAGS.test_max_length * 2 / 3, sess)
            print
            task.run(ntm, FLAGS.test_max_length * 3 / 3, sess)
        elif FLAGS.task.startswith('brainfuck'):
            from tasks.brainfuck_common import context_products

            correct = 0
            count = 0
            for idx in range(len(context_products)):
                seq_in, seq_out, outputs, rounded_outputs, read_ws, write_ws, loss = task.run(ntm, FLAGS.test_max_length, sess, idx=idx, print_=True)
                count += 1
                # if (seq_out[0] == rounded_outputs[0]).sum() == FLAGS.output_dim:
                if task.seq_to_inst(seq_out, seq_out) == task.seq_to_inst(rounded_outputs, outputs):
                    correct += 1
            print('correct: %d / count: %d' % (correct, count))
        else:
            task.run(ntm, FLAGS.test_max_length, sess)


if __name__ == '__main__':
    tf.app.run()
