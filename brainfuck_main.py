from __future__ import absolute_import

import importlib
import tensorflow as tf
from ntm_cell import NTMCell
from ntm import NTM

flags = tf.app.flags
flags.DEFINE_string("task", "copy", "Task to run [copy, recall]")
flags.DEFINE_integer("epoch", 100000, "Epoch to train [100000]")
flags.DEFINE_integer("input_dim", 10, "Dimension of input [10]")
flags.DEFINE_integer("output_dim", 10, "Dimension of output [10]")
flags.DEFINE_integer("min_length", 1, "Minimum length of input sequence [1]")
flags.DEFINE_integer("max_length", 10, "Maximum length of output sequence [10]")
flags.DEFINE_integer("controller_layer_size", 2, "The size of LSTM controller [1]")
flags.DEFINE_integer("write_head_size", 1, "The number of write head [1]")
flags.DEFINE_integer("read_head_size", 1, "The number of read head [1]")
flags.DEFINE_integer("test_max_length", 120, "Maximum length of output sequence [120]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS


def create_ntm(FLAGS, sess, **ntm_args):
    cell = NTMCell(
        input_dim=FLAGS.input_dim,
        output_dim=FLAGS.output_dim,
        controller_layer_size=FLAGS.controller_layer_size,
        write_head_size=FLAGS.write_head_size,
        read_head_size=FLAGS.read_head_size)
    ntm = NTM(
        cell, sess, FLAGS.min_length, FLAGS.max_length,
        test_max_length=FLAGS.test_max_length, scope='NTM-%s' % FLAGS.task, **ntm_args)
    return cell, ntm


class Task(object):

    def __init__(self, session, name):
        self.session = session
        self.name = name

    def load(self):
        self.preset_flags(FLAGS)
        self.cell, self.ntm = create_ntm(FLAGS, self.session, forward_only=True)
        self.ntm.load(FLAGS.checkpoint_dir, self.module_name)

    @property
    def module_name(self):
        return 'brainfuck_part_%s' % self.name

    @property
    def module(self):
        try:
            module = importlib.import_module('tasks.%s' % self.module_name)
        except ImportError:
            print("task '%s' does not have implementation" % self.module_name)
            raise
        return module

    def preset_flags(self, FLAGS):
        FLAGS.task = self.module_name
        self.module.preset_flags(FLAGS)


def main(_):
    from tasks.brainfuck_common import context_products
    from brainfuck.core import Instruction

    with tf.device('/cpu:0'), tf.Session() as sess:
        tasks = [Task(sess, name) for name in ['head', 'value', 'skip', 'direction', 'interaction']]
        for task in tasks:
            task.preset_flags(FLAGS)
            task.load()

        correct = 0
        count = 0
        for idx in range(len(context_products)):
            l_expected = []
            l_actual = []
            for i, task in enumerate(tasks):
                task.preset_flags(FLAGS)
                seq_in, seq_out, outputs, rounded_outputs, read_ws, write_ws, loss = task.module.run(task.ntm, FLAGS.test_max_length, sess, idx=idx, print_=False)
                p_expected = task.module.seq_to_inst(seq_out, seq_out)
                l_expected.append(p_expected[i])
                p_actual = task.module.seq_to_inst(rounded_outputs, outputs)
                l_actual.append(p_actual[i])
            expected = Instruction(*l_expected)
            actual = Instruction(*l_actual)
            count += 1
            # if (seq_out[0] == rounded_outputs[0]).sum() == FLAGS.output_dim:
            if expected == actual:
                correct += 1
            else:
                print('expected', expected)
                print('actual', actual)
        print('correct: %d / count: %d' % (correct, count))


if __name__ == '__main__':
    tf.app.run()
