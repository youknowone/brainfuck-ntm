from __future__ import absolute_import

import os
import time
import numpy as np
import tensorflow as tf
import itertools
from random import randint

from ntm import NTM
from ntm_cell import NTMCell
from brainfuck.brainfuck import TableTranslator
from brainfuck.core import State


print_interval = 5

translator = TableTranslator()
context_products = []
for context in itertools.product('+-><[],.', [True, False], [0, 1, 2], [1, -1]):
    try:
        instruction = translator.translate(context[0], State(*context[1:]))
    except KeyError:
        print('unacceptable context:', context)
    else:
        context_products.append((context, instruction))


def create_ntm(FLAGS, sess, forward_only, **ntm_args):
    cell = NTMCell(
        input_dim=FLAGS.input_dim,
        output_dim=FLAGS.output_dim,
        controller_layer_size=FLAGS.controller_layer_size,
        write_head_size=FLAGS.write_head_size,
        read_head_size=FLAGS.read_head_size)
    ntm = NTM(
        cell, sess, FLAGS.min_length, FLAGS.max_length,
        test_max_length=FLAGS.test_max_length, forward_only=forward_only, **ntm_args)
    return cell, ntm


def meta_run(ntm, seq_length, sess, generate_training_sequence, idx=None, seq_to_inst=None, print_=True, **ntm_args):
    context, instruction, seq_in, seq_out = generate_training_sequence(seq_length, ntm.cell, idx=idx)

    feed_dict = {input_: vec for vec, input_ in zip(seq_in, ntm.inputs)}
    feed_dict.update(
        {true_output: vec for vec, true_output in zip(seq_out, ntm.true_outputs)}
    )
    if ntm_args.get('is_copy', True):
        start_symbol = np.zeros([ntm.cell.input_dim], dtype=np.float32)
        start_symbol[0] = 1
        end_symbol = np.zeros([ntm.cell.input_dim], dtype=np.float32)
        end_symbol[1] = 1
        feed_dict.update({
            ntm.start_symbol: start_symbol,
            ntm.end_symbol: end_symbol
        })

    input_states = [state['write_w'][0] for state in ntm.input_states[seq_length]]
    output_states = [state['read_w'][0] for state in ntm.get_output_states(seq_length)]

    result = sess.run(ntm.get_outputs(seq_length) +
                      input_states + output_states +
                      [ntm.get_loss(seq_length)],
                      feed_dict=feed_dict)

    is_sz = len(input_states)
    os_sz = len(output_states)

    outputs = result[:seq_length]
    read_ws = result[seq_length:seq_length + is_sz]
    write_ws = result[seq_length + is_sz:seq_length + is_sz + os_sz]
    loss = result[-1]

    rounded_outputs = np.round(outputs)
    if print_:
        np.set_printoptions(suppress=True)
        correct = (seq_out[0] == rounded_outputs[0]).sum() == seq_out[0].size
        if not correct:
            print(" -- %s" % ('CORRECT' if correct else 'incorrect'))
            print(" input : ")
            print(context)
            print(" true output : ")
            print(instruction)
            print(seq_to_inst(seq_out))
            print(" predicted output :")
            print(seq_to_inst(rounded_outputs))
            print(" Loss : %f" % loss)
        np.set_printoptions(suppress=False)
    return seq_in, seq_out, outputs, rounded_outputs, read_ws, write_ws, loss


def meta_train(ntm, config, sess, generate_training_sequence, **ntm_args):
    if not os.path.isdir(config.checkpoint_dir):
        raise Exception(" [!] Directory %s not found" % config.checkpoint_dir)

    is_copy = ntm_args.get('is_copy', True)
    if is_copy:
        # delimiter flag for start and end
        start_symbol = np.zeros([config.input_dim], dtype=np.float32)
        start_symbol[0] = 1
        end_symbol = np.zeros([config.input_dim], dtype=np.float32)
        end_symbol[1] = 1

    print(" [*] Initialize all variables")
    tf.initialize_all_variables().run()
    print(" [*] Initialization finished")

    start_time = time.time()
    for idx in xrange(config.epoch):
        seq_length = randint(config.min_length, config.max_length)
        context, instruction, seq_in, seq_out = generate_training_sequence(seq_length, config)

        input_dict = {input_: vec for vec, input_ in zip(seq_in, ntm.inputs)}
        output_dict = {true_output: vec for vec, true_output in zip(seq_out, ntm.true_outputs)}
        feed_dict = input_dict
        feed_dict.update(output_dict)
        if is_copy:
            feed_dict.update({
                ntm.start_symbol: start_symbol,
                ntm.end_symbol: end_symbol
            })

        _, cost, step = sess.run([ntm.optims[seq_length],
                                  ntm.get_loss(seq_length),
                                  ntm.global_step], feed_dict=feed_dict)

        if idx % 100 == 0:
            ntm.save(config.checkpoint_dir, config.task, step)

        if idx % print_interval == 0:
            print("[%5d] %2d: %.2f (%.1fs)"
                  % (idx, seq_length, cost, time.time() - start_time))

    print("Training Brainfuck task finished")
