import os, sys
import subprocess, argparse
import numpy as np

from deepcrime_scripts.utils import *
from deepcrime_scripts.stats import is_diff_sts_classification

########################################
# Note that this script is for checking the mutants killed by train set 
# as well as test set (i.e., Not only by test set).
# I used the set of mutants killed by train set, and then checked those mutants
# are killed by test set or not.
########################################

DMPP_HOME_PATH = '/home/DC_replication/DCReplication/deepmutationpp'


def get_prediction_array_dmpp_strong_testset(model, name_prefix, model_dir):       
    model_file = os.path.join(model_dir, name_prefix + ".h5")
    return model.get_prediction_info_strong_testset(model_file) #


def get_prediction_array_dmpp_weak_testset(model, name_prefix, model_dir):       
    model_file = os.path.join(model_dir, name_prefix + ".h5")    
    weak_test_path = '/home/DC_replication/Datasets/Audio/'
    return model.get_prediction_info_weak_testset(model_file, weak_test_path) #


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--op_no", type=int, help="Specify the operation number. 0 to 7")    
    parser.add_argument("--test_type", type=str, default="strong", help="Either strong or weak tests.")    
    args = parser.parse_args()
    op_no = args.op_no    

    # Configurable arguments (but fixed for the study)
    ratio = 0.05
    total_mutant_num = 20
    tatal_model_training = 20 
    standard_deviation = 0.5

    subject_name = 'audio'
    base_model = get_subject_model(subject_name)
    orig_model_dir = f'{DMPP_HOME_PATH}/original_models/multiple_training/'
    if args.test_type == 'strong':
        base_model_accs_path = f'{orig_model_dir}/{subject_name}_original_acc_strong_testset.txt'
    else:
        base_model_accs_path = f'{orig_model_dir}/{subject_name}_original_acc_weak_testset.txt'

    # Get original accs
    orig_accs = []
    if not os.path.exists(base_model_accs_path):
        with open(base_model_accs_path, 'w') as f:
            # Iterate original trained models to get their accs
            for model_no in range(tatal_model_training):
                if args.test_type == 'strong':
                    orig_prediction_info = get_prediction_array_dmpp_strong_testset(base_model, f'{subject_name}_original_{model_no}', orig_model_dir)        
                else:
                    orig_prediction_info = get_prediction_array_dmpp_weak_testset(base_model, f'{subject_name}_original_{model_no}', orig_model_dir)
                orig_acc = len(np.argwhere(orig_prediction_info == True)) / len(orig_prediction_info) * 100
                orig_accs.append(orig_acc)
                f.write(f'{model_no},{orig_acc:.3f}\n')
    else:
        with open(base_model_accs_path, 'r') as f:
            for line in f:
                model_no, orig_acc = line.strip().split(',')
                orig_acc = float(orig_acc)                
                orig_accs.append(orig_acc)

    for mut_no in range(1, total_mutant_num+1): # mut_no starts from 1
        mutant_accs = []
        for model_no in range(tatal_model_training):
            mutants_path = f'{DMPP_HOME_PATH}/dmpp_mutants_{subject_name}/mutated_models_{subject_name}_trained:{model_no}/{op_no}'
            for path in sorted(os.listdir(mutants_path)):
                if path.endswith(f"_{mut_no}.h5"):
                    if args.test_type == 'strong':
                        mutant_prediction_info = get_prediction_array_dmpp_strong_testset(base_model, path[:-3], mutants_path)
                    else:
                        mutant_prediction_info = get_prediction_array_dmpp_weak_testset(base_model, path[:-3], mutants_path)
                    mutant_acc = len(np.argwhere(mutant_prediction_info == True)) / len(mutant_prediction_info) * 100
                    mutant_accs.append(mutant_acc)
                    break

        if len(mutant_accs) != tatal_model_training:
            continue

        print(f'Original accs: {orig_accs}')
        print(f'Mutant {mut_no} accs: {mutant_accs}')        
        is_sts, p_value, effect_size = is_diff_sts_classification(orig_accs, mutant_accs)
        
        if args.test_type == 'strong':
            output_acc_path = f'./output/mutants_accs_strong_testset.csv'
        else:
            output_acc_path = f'./output/mutants_accs_weak_testset.csv'

        with open(output_acc_path, 'a') as f:
            for model_no in range(tatal_model_training):
                f.write(f'{op_no},{mut_no},{model_no},{mutant_accs[model_no]:.3f},{is_sts}\n')