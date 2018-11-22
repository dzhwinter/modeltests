import os, sys, time
import functools
import numpy as np
import math
import argparse
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.optimizer as optimizer
from paddle.fluid.framework import Program, program_guard
from paddle.fluid.transpiler import memory_optimize
from utility import add_arguments, print_arguments
import paddle.dataset.flowers as flowers
from paddle.dataset.flowers import *

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',       int,   32,                  "Minibatch size.")
add_arg('with_ir_mem_opt',  bool,  True,                 "Whether to use ir memory optimization or not.")
add_arg('model',  str,  "fc_net",                 "Whether to use ir memory optimization or not.")


def cascade_net(y, word):
    STEPS = 3
    emb = [
        fluid.layers.embedding(
            word, size=[65536, 256], param_attr='emb') for _ in range(STEPS)
    ]

    left = emb.pop(0)
    while len(emb) != 0:
        right = emb.pop(0)
        left = fluid.layers.sums([left, right])
    # mean = fluid.layers.mean(left)
    y_predict = layers.fc(left, size=10, act=None)
    cost = layers.square_error_cost(input=y_predict, label=y)
    avg_cost = layers.mean(cost)
    opt = optimizer.SGD(learning_rate=0.001)
    opt = opt.minimize(avg_cost)
    return avg_cost

def fake_reader(batch_size=32):
    def reader():
        while True:
            y = np.random.randint(low=0, high=10, size=(batch_size, 1))
            word = np.random.randint(low=0, high=10, size=(batch_size, 1))
            yield y, word
    return reader

def train(args):
    word = fluid.layers.data(name='word', shape=[1], dtype='int64')
    y = layers.data(name='y', shape=[1], dtype='float32')

    avg_cost = cascade_net(y, word)

    if args.with_ir_mem_opt:
        build_strategy = fluid.BuildStrategy()
        build_strategy.memory_optimize = True
        build_strategy.debug_graphviz_path = "./debug"
        train_exe = fluid.ParallelExecutor(
            use_cuda=False, loss_name=avg_cost.name, build_strategy=build_strategy)
        train_batch_size = args.batch_size
        import random
        random.seed(1000)
        np.random.seed(1000)
	place = fluid.CPUPlace()
	exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        # train_reader = paddle.batch(
        #     fake_reader(batch_size=train_batch_size), batch_size=train_batch_size)
        train_reader = fake_reader(batch_size = train_batch_size)
        # train_reader = paddle.batch(
        #     flowers.test(use_xmap=False), batch_size=train_batch_size)
        feeder = fluid.DataFeeder(place=place, feed_list=[y, word])
        fetch_list = [avg_cost.name]
	with open("program.txt", "w") as f:
	    f.write(str(fluid.default_main_program()))
        for batch_id, data in enumerate(train_reader()):
            t1 = time.time()
            loss = train_exe.run(fetch_list, feed=feeder.feed(data))
            t2 = time.time()
            period = t2 - t1
            loss = np.mean(np.array(loss))
            if batch_id % 10 == 0:
                print("Pass {0}, trainbatch {1}, loss {2}, \
                       time {5}"
                                                   .format(pass_id, \
                       batch_id, loss, \
                       "%2.2f sec" % period))
                sys.stdout.flush()
    else:
        program = avg_cost.block.program
        fluid.memory_optimize(program, print_log=True, level=1)

    # place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    # exe = fluid.Executor(place)
    # exe.run(fluid.default_startup_program())

def main():
    args = parser.parse_args()
    print_arguments(args)
    train(args)

if __name__ == '__main__':
    main()
