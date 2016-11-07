from __future__ import absolute_import

import numpy as np
import random

from brainfuck.core import Instruction, State

from . import brainfuck_common as common

print_interval = 5


def preset_flags(FLAGS):
    FLAGS.controller_layer_size = 2
    FLAGS.max_length = 1
    FLAGS.test_max_length = 1
    FLAGS.input_dim = 2 + 8 + 1 + 3 + 1  # 13 + 2 code 8 nonzero 1 skip 3 direction 1
    FLAGS.output_dim = 2 + 3  # 3 + 3 + 3 + 1 + 3 + 2  # 13 + 2 head 3 value 3 skip 3 direction 1 interaction 3


def run(ntm, seq_length, sess, idx=None, print_=True):
    return common.meta_run(ntm, seq_length, sess, generate_training_sequence=generate_training_sequence, idx=idx, seq_to_inst=seq_to_inst, print_=print_)


def train(ntm, config, sess, **kw):
    return common.meta_train(ntm, config, sess, generate_training_sequence=generate_training_sequence, **kw)


tf_set = [False, True]
order = '><+-.,[]'


def generate_training_sequence(length, config, idx=None):

    if not idx:
        context, instruction = random.choice(common.context_products)
    else:
        context, instruction = common.context_products[idx % len(common.context_products)]

    token = context[0]
    if token is None:
        token = random.choice(order)
    token_id = order.index(token)
    nonzero = context[1]
    if nonzero is None:
        nonzero = random.choice(tf_set)
    skip = context[2]
    if skip is None:
        skip = random.choice((0, 1, 2))
    direction = context[3]
    direction_id = direction > 0

    seq_in = np.zeros([1, config.input_dim], dtype=np.float32)
    # marker 2 bits
    seq_in[0][2 + token_id] = 1
    # token 8 bits -> 10
    seq_in[0][10] = nonzero
    # nonzero 1 bit -> 11
    seq_in[0][11 + skip] = 1
    # skip 3 bits -> 14
    seq_in[0][14] = direction_id

    direction_id = instruction.direction > 0
    interaction_id = [None, 'i', 'o'].index(instruction.interaction)

    seq_out = np.zeros([1, config.output_dim], dtype=np.float32)
    bits = seq_out[0]
    bits[2 + instruction.skip_diff + 1] = 1
    # bits[2 + instruction.head_diff + 1] = 1
    # bits[5 + instruction.value_diff + 1] = 1
    # bits[8 + instruction.skip_diff + 1] = 1
    # bits[11] = direction_id
    # bits[12 + interaction_id] = 1
    return (token, nonzero, skip, direction), instruction, list(seq_in), list(seq_out)


def seq_to_inst(seq, real):
    bits = seq[0]
    values = real[0]
    diff_value = max(values[2], values[3], values[4])
    skip_diff = 0 if values[3] == diff_value else (values[2] == diff_value and -1) or (values[4] == diff_value and 1) or False
    return Instruction(0, 0, skip_diff, 0, None)

    head_diff = -1 if bits[2] else 1 if bits[4] else 0
    value_diff = -1 if bits[5] else 1 if bits[7] else 0
    skip_diff = -1 if bits[8] else 1 if bits[10] else 0
    interaction = (bits[13] and 'i') or (bits[14] and 'o') or None
    instruction = Instruction(head_diff, value_diff, skip_diff, direction, interaction)
    return instruction
