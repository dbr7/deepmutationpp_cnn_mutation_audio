import os
import csv
import h5py
import glob
from deepcrime_scripts.Model import *
import statsmodels.stats.proportion as smp


def get_outcome_row_index(filename):
    if ('disable_batching' in filename) or ('remove_validation_set' in filename):
        return 2
    else:
        return 3


def get_list_of_files_by_name(name_prefix, dir):
    files = []
    # r=root, d=directories, f=files
    for r, d, f in os.walk(dir):
        for file in f:
            if name_prefix in file:
                files.append(os.path.join(r, file))


def get_prefix_list(directory):
    prefix_list = set()
    for filename in glob.glob((os.path.join(directory, '*'))):
        if 'original' not in filename and '_ki.npy' in filename:
            end_index = filename.rfind('_')
            prefix_list.add(filename[0:end_index].replace(directory + os.path.sep, ''))

    return prefix_list


def is_binary_search_operator(filename):
    operator_list = ['change_label', 'delete_td', 'unbalance_td', 'add_noise',
                     'output_classes_overlap', 'change_epochs', 'change_learning_rate', 'change_patience']
    for operator in operator_list:
        if operator in str(filename):
            return True

    return False


def get_subject_model(subject_name):
    if subject_name == "mnist":
        return MnistModel()
    elif subject_name == "movie_recomm":
        return MovieModel()
    elif subject_name == "lenet":
        return UnityModel()
    elif subject_name == "udacity":
        return UdacityModel()
    elif subject_name == "audio":
        return AudioModel()


def get_confidence_intervals():
    confidence_intervals = []

    for i in range(0, 21):
        (low, high) = smp.proportion_confint(i, 20, alpha=0.10, method='wilson')
        confidence_intervals.append((low, high))

    return confidence_intervals


def write_list_to_csv(csv_file, value_list):
    with open(csv_file, 'a') as f1:
        writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
        for node in value_list:
            writer.writerow([node])
