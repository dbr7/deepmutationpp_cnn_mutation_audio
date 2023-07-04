from deepmutationpp.cnn_mutation_audio.src.cnn_operator import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import argparse
from deepmutationpp.cnn_mutation_audio.src.utils import summary_model, color_preprocessing, model_predict
from termcolor import colored
import gc
import time
import numpy as np
import keras.backend as K
from progressbar import *
# from audio_model import test_ds


def cnn_mutants_generation(ori_model, operator, ratio, standard_deviation=0.5):
    """

    :param ori_model:
    :param operator:
    :param ratio:
    :param standard_deviation:
    :return:
    """
    if operator < 5:
        cnn_operator(ori_model, operator, ratio, standard_deviation)
    else:
        new_model = cnn_operator(ori_model, operator, ratio, standard_deviation)
        return new_model
    return ori_model


def generator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-model_path",
                        type=str,
                        help="ori model path")
    parser.add_argument("--data_type", "-data_type",
                        type=str,
                        help="mnist or cifar-10")
    parser.add_argument("--operator", "-operator",
                        type=int,
                        help="mutator")
    parser.add_argument("--ratio", "-ratio",
                        type=float,
                        help="mutation ratio")
    parser.add_argument("--save_path", "-save_path",
                        type=str,
                        help="mutants save path")
    parser.add_argument("--threshold", "-threshold",
                        type=float,
                        default=0.9,
                        help="ori acc * threshold must > mutants acc")
    parser.add_argument("--num", "-num",
                        type=int,
                        default=1,
                        help="mutants number")
    parser.add_argument("--standard_deviation", "-standard_deviation",
                        type=float,
                        default=0.5,
                        help="standard deviation for gaussian fuzzing")
    args = parser.parse_args()
    model_path = args.model_path
    data_type = args.data_type
    operator = args.operator
    ratio = args.ratio
    save_path = args.save_path
    threshold = args.threshold
    num = args.num
    standard_deviation = args.standard_deviation

    # load data
    # if data_type == 'audio':

    model = load_model(model_path, compile=False)
    # ori_acc = model_predict(model, test_ds)
    # threshold = ori_acc * threshold
    weight_count, neuron_count, weights_dict, neuron_dict = summary_model(model)
    print(colored("operator: %s" % cnn_operator_name(operator), 'blue'))
    # print(colored("ori acc: %f" % ori_acc, 'blue'))
    # print(colored("threshold acc: %f" % threshold, 'blue'))
    if operator == 0 or operator == 1:
        print("total weights: ", weight_count)
        print("process weights num: ", int(weight_count * ratio) if int(weight_count * ratio) > 0 else 1)
    elif 2 <= operator <= 4:
        print("total neuron: ", neuron_count)
        print("process neuron num: ", int(neuron_count * ratio) if int(neuron_count * ratio) > 0 else 1)

    # mutants generation
    p_bar = ProgressBar().start()
    i = 1
    start_time = time.process_time()
    while i <= num:
        print("Start")
        if i != 1:
            model = load_model(model_path, compile=False)
            print("Tuta")
        print(i)
        print(model)
        new_model = cnn_mutants_generation(model, operator, ratio, standard_deviation)
        # new_acc = model_predict(new_model, test_ds)
        # if new_acc < threshold:
        #     K.clear_session()
        #     del model
        #     del new_model
        #     gc.collect()
        #     continue
        final_path = save_path + "/" + cnn_operator_name(operator) + "_" + str(ratio) + "_" + str(i) + ".h5"
        new_model.save(final_path)
        p_bar.update(int((i / num) * 100))
        i += 1
        K.clear_session()
        del model
        del new_model
        gc.collect()
        print("End")
    p_bar.finish()
    elapsed = (time.process_time() - start_time)
    print("running time: ", elapsed)


if __name__ == '__main__':
    generator()

# python generator.py --model_path ../../models/mnist_lenet5.h5 --operator 0 --ratio 0.01 --save_path ../../mutants --num 2

