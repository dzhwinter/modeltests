from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import time
import sys
import functools
import math
import paddle
import paddle.fluid as fluid
import paddle.dataset.flowers as flowers
from paddle.dataset.flowers import *
import models
import reader
import argparse
from models.learning_rate import cosine_decay
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,   256,                  "Minibatch size.")
add_arg('use_gpu',          bool,  True,                 "Whether to use GPU or not.")
add_arg('total_images',     int,   1281167,              "Training image number.")
add_arg('num_epochs',       int,   120,                  "number of epochs.")
add_arg('class_dim',        int,   1000,                 "Class number.")
add_arg('image_shape',      str,   "3,224,224",          "input image size")
add_arg('model_save_dir',   str,   "output",             "model save directory")
add_arg('with_mem_opt',     bool,  True,                 "Whether to use memory optimization or not.")
add_arg('with_ir_mem_opt',  bool,  True,                 "Whether to use ir memory optimization or not.")
add_arg('pretrained_model', str,   None,                 "Whether to use pretrained model.")
add_arg('checkpoint',       str,   None,                 "Whether to resume checkpoint.")
add_arg('lr',               float, 0.1,                  "set learning rate.")
add_arg('lr_strategy',      str,   "piecewise_decay",    "Set the learning rate decay strategy.")
add_arg('model',            str,   "SE_ResNeXt50_32x4d", "Set the network to use.")
add_arg('enable_ce',        bool,  False,                "If set True, enable continuous evaluation job.")
add_arg('data_dir',         str,   "./data/ILSVRC2012",  "The ImageNet dataset root dir.")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]


def optimizer_setting(params):
    ls = params["learning_strategy"]

    if ls["name"] == "piecewise_decay":
        if "total_images" not in params:
            total_images = 1281167
        else:
            total_images = params["total_images"]

        batch_size = ls["batch_size"]
        step = int(total_images / batch_size + 1)

        bd = [step * e for e in ls["epochs"]]
        base_lr = params["lr"]
        lr = []
        lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))
    elif ls["name"] == "cosine_decay":
        if "total_images" not in params:
            total_images = 1281167
        else:
            total_images = params["total_images"]

        batch_size = ls["batch_size"]
        step = int(total_images / batch_size + 1)

        lr = params["lr"]
        num_epochs = params["num_epochs"]

        optimizer = fluid.optimizer.Momentum(
            learning_rate=cosine_decay(
                learning_rate=lr, step_each_epoch=step, epochs=num_epochs),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))
    else:
        lr = params["lr"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=lr,
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))

    return optimizer


def train(args):
    # parameters from arguments
    fluid.default_startup_program().random_seed = 1000
    os.environ['CPU_NUM'] = str(8)
    class_dim = args.class_dim
    model_name = args.model
    checkpoint = args.checkpoint
    pretrained_model = args.pretrained_model
    with_memory_optimization = args.with_mem_opt
    model_save_dir = args.model_save_dir
    image_shape = [int(m) for m in args.image_shape.split(",")]

    assert model_name in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    # model definition
    model = models.__dict__[model_name]()

    if args.enable_ce:
        assert model_name == "SE_ResNeXt50_32x4d"
        fluid.default_startup_program().random_seed = 1000
        model.params["dropout_seed"] = 100
        class_dim = 102

    if model_name == "GoogleNet":
        out0, out1, out2 = model.net(input=image, class_dim=class_dim)
        cost0 = fluid.layers.cross_entropy(input=out0, label=label)
        cost1 = fluid.layers.cross_entropy(input=out1, label=label)
        cost2 = fluid.layers.cross_entropy(input=out2, label=label)
        avg_cost0 = fluid.layers.mean(x=cost0)
        avg_cost1 = fluid.layers.mean(x=cost1)
        avg_cost2 = fluid.layers.mean(x=cost2)

        avg_cost = avg_cost0 + 0.3 * avg_cost1 + 0.3 * avg_cost2
        acc_top1 = fluid.layers.accuracy(input=out0, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out0, label=label, k=5)
    else:
        out = model.net(input=image, class_dim=class_dim)
        cost = fluid.layers.cross_entropy(input=out, label=label)

        avg_cost = fluid.layers.mean(x=cost)
        acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)

    test_program = fluid.default_main_program().clone(for_test=True)

    # parameters from model and arguments
    params = model.params
    params["total_images"] = args.total_images
    params["lr"] = args.lr
    params["num_epochs"] = args.num_epochs
    params["learning_strategy"]["batch_size"] = args.batch_size
    params["learning_strategy"]["name"] = args.lr_strategy
    # set random
    model.params["dropout_seed"] = 1000

    # initialize optimizer
    optimizer = optimizer_setting(params)
    opts = optimizer.minimize(avg_cost)

    print("before optimize")
    lower_usage, upper_usage, unit = fluid.contrib.memory_usage(
        fluid.default_main_program(), batch_size=args.batch_size)
    print("memory usage is about %.3f - %.3f %s" %
          (lower_usage, upper_usage, unit))
    if with_memory_optimization:
        fluid.memory_optimize(fluid.default_main_program(), level=1, print_log=True)
    print("after optimize")
    lower_usage, upper_usage, unit = fluid.contrib.memory_usage(
        fluid.default_main_program(), batch_size=args.batch_size)
    print("memory usage is about %.3f - %.3f %s" %
          (lower_usage, upper_usage, unit))

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    with open("big_model.txt", "w") as f:
	f.write(str(fluid.default_main_program()))

    if checkpoint is not None:
        fluid.io.load_persistables(exe, checkpoint)

    if pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    train_batch_size = args.batch_size
    test_batch_size = 16

    # if not args.enable_ce:
    #     train_reader = paddle.batch(reader.train(), batch_size=train_batch_size)
    #     test_reader = paddle.batch(reader.val(), batch_size=test_batch_size)
    # else:
    # use flowers dataset for CE and set use_xmap False to avoid disorder data
    # but it is time consuming. For faster speed, need another dataset.
    import random
    random.seed(1000)
    np.random.seed(1000)
    # train_reader = paddle.batch(
    #     flowers.test(use_xmap=False), batch_size=train_batch_size)
    train_reader = paddle.batch(
        flowers.train(use_xmap=True), batch_size=train_batch_size)
    test_reader = paddle.batch(
        flowers.test(use_xmap=False), batch_size=test_batch_size)

    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    build_strategy = fluid.BuildStrategy()
    if args.with_ir_mem_opt:
	build_strategy.memory_optimize = True
	build_strategy.debug_graphviz_path = "./"
    train_exe = fluid.ParallelExecutor(
        use_cuda=True if args.use_gpu else False, loss_name=avg_cost.name, build_strategy=build_strategy)

    fetch_list = [avg_cost.name, acc_top1.name, acc_top5.name]

    gpu = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    gpu_nums = len(gpu.split(","))
    for pass_id in range(params["num_epochs"]):
        train_info = [[], [], []]
        test_info = [[], [], []]
        train_time = []
        for batch_id, data in enumerate(train_reader()):
	    #print(data)
            t1 = time.time()
            loss, acc1, acc5 = train_exe.run(fetch_list, feed=feeder.feed(data))
            t2 = time.time()
            period = t2 - t1
            loss = np.mean(np.array(loss))
            acc1 = np.mean(np.array(acc1))
            acc5 = np.mean(np.array(acc5))
            train_info[0].append(loss)
            train_info[1].append(acc1)
            train_info[2].append(acc5)
            train_time.append(period)
            if batch_id % 10 == 0:
                print("Pass {0}, trainbatch {1}, loss {2}, \
                       acc1 {3}, acc5 {4} time {5}"
                                                   .format(pass_id, \
                       batch_id, loss, acc1, acc5, \
                       "%2.2f sec" % period))
                sys.stdout.flush()

        train_loss = np.array(train_info[0]).mean()
        train_acc1 = np.array(train_info[1]).mean()
        train_acc5 = np.array(train_info[2]).mean()
        train_speed = np.array(train_time).mean() / train_batch_size
        cnt = 0
        for test_batch_id, data in enumerate(test_reader()):
            t1 = time.time()
            loss, acc1, acc5 = exe.run(test_program,
                                       fetch_list=fetch_list,
                                       feed=feeder.feed(data))
            t2 = time.time()
            period = t2 - t1
            loss = np.mean(loss)
            acc1 = np.mean(acc1)
            acc5 = np.mean(acc5)
            test_info[0].append(loss * len(data))
            test_info[1].append(acc1 * len(data))
            test_info[2].append(acc5 * len(data))
            cnt += len(data)
            if test_batch_id % 10 == 0:
                print("Pass {0},testbatch {1},loss {2}, \
                       acc1 {3},acc5 {4},time {5}"
                                                  .format(pass_id, \
                       test_batch_id, loss, acc1, acc5, \
                       "%2.2f sec" % period))
                sys.stdout.flush()

        test_loss = np.sum(test_info[0]) / cnt
        test_acc1 = np.sum(test_info[1]) / cnt
        test_acc5 = np.sum(test_info[2]) / cnt

        print("End pass {0}, train_loss {1}, train_acc1 {2}, train_acc5 {3}, "
              "test_loss {4}, test_acc1 {5}, test_acc5 {6}".format(pass_id, \
              train_loss, train_acc1, train_acc5, test_loss, test_acc1, \
              test_acc5))
        sys.stdout.flush()

        model_path = os.path.join(model_save_dir + '/' + model_name,
                                  str(pass_id))
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        fluid.io.save_persistables(exe, model_path)

        # This is for continuous evaluation only
        if args.enable_ce and pass_id == args.num_epochs - 1:
            if gpu_nums == 1:
                # Use the mean cost/acc for training
                print("kpis	train_cost	%s" % train_loss)
                print("kpis	train_acc_top1	%s" % train_acc1)
                print("kpis	train_acc_top5	%s" % train_acc5)
                # Use the mean cost/acc for testing
                print("kpis	test_cost	%s" % test_loss)
                print("kpis	test_acc_top1	%s" % test_acc1)
                print("kpis	test_acc_top5	%s" % test_acc5)
                print("kpis	train_speed	%s" % train_speed)
            else:
                # Use the mean cost/acc for training
                print("kpis	train_cost_card%s	%s" % (gpu_nums, train_loss))
                print("kpis	train_acc_top1_card%s	%s" % (gpu_nums, train_acc1))
                print("kpis	train_acc_top5_card%s	%s" % (gpu_nums, train_acc5))
                # Use the mean cost/acc for testing
                print("kpis	test_cost_card%s	%s" % (gpu_nums, test_loss))
                print("kpis	test_acc_top1_card%s	%s" % (gpu_nums, test_acc1))
                print("kpis	test_acc_top5_card%s	%s" % (gpu_nums, test_acc5))
                print("kpis	train_speed_card%s	%s" % (gpu_nums, train_speed))


def main():
    args = parser.parse_args()
    print_arguments(args)
    train(args)


if __name__ == '__main__':
    main()