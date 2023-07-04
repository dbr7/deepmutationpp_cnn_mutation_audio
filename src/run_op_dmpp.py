import os, sys
import subprocess, argparse
import numpy as np
from deepcrime_scripts.utils import *
from deepcrime_scripts.stats import is_diff_sts_classification

DMPP_HOME_PATH = '/home/DC_replication/DCReplication/deepmutationpp'


def get_prediction_array_dmpp(model, name_prefix, model_dir):       
    model_file = os.path.join(model_dir, name_prefix + ".h5")
    return model.get_prediction_info(model_file)


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--op_no", type=int, help="Specify the operation number. 0 to 7")
    parser.add_argument("--gpu", action="store_true", help="Specify if you want to run on GPU")
    args = parser.parse_args()

    # This script is for generating mutants per mutation operator.
    op_no = args.op_no
    if args.gpu:
        gpu_no = int(op_no) % 4

    # Configurable arguments (but fixed for the study)
    ratio = 0.05
    total_mutant_num = 20
    tatal_model_training = 20 
    standard_deviation = 0.5

    subject_name = 'audio' #
    base_model = get_subject_model(subject_name)
    orig_model_dir = f'{DMPP_HOME_PATH}/original_models/multiple_training/'
    base_model_accs_path = f'{orig_model_dir}/{subject_name}_original_acc.txt'

    # Get original accs
    orig_accs = []
    if not os.path.exists(base_model_accs_path):
        with open(base_model_accs_path, 'w') as f:
            # Iterate original trained models to get their accs
            for model_no in range(tatal_model_training):
                orig_prediction_info = get_prediction_array_dmpp(base_model, f'{subject_name}_original_{model_no}', orig_model_dir)        
                orig_acc = len(np.argwhere(orig_prediction_info == True)) / len(orig_prediction_info) * 100
                orig_accs.append(orig_acc)
                print(f'Original model {model_no} acc: {orig_acc:.3f}')
                f.write(f'{model_no},{orig_acc:.3f}\n')
    else:
        with open(base_model_accs_path, 'r') as f:
            for line in f:
                model_no, orig_acc = line.strip().split(',')
                orig_acc = float(orig_acc)                
                orig_accs.append(orig_acc)

    # Generate mutants
    for model_no in range(tatal_model_training): 
        # The mutants will be saved to "mutants_path"
        mutants_path = f'{DMPP_HOME_PATH}/mutated_models_{subject_name}_trained:{model_no}/{op_no}'
        orig_model_path = f'{orig_model_dir}/{subject_name}_original_{model_no}.h5'

        if not os.path.exists(mutants_path):
            os.makedirs(mutants_path)    

        # Run generator.py
        if args.gpu:
            bash_command = f"CUDA_VISIBLE_DEVICES={gpu_no} python generator.py -model_path {orig_model_path} -data_type {subject_name} -operator {op_no} -ratio {ratio} -save_path {mutants_path} -num {total_mutant_num} -standard_deviation {standard_deviation}"
        else:
            bash_command = f"python generator.py -model_path {orig_model_path} -data_type {subject_name} -operator {op_no} -ratio {ratio} -save_path {mutants_path} -num {total_mutant_num} -standard_deviation {standard_deviation}"

        print(bash_command)
        subprocess.call(bash_command, shell=True)
    
    # Collect mutants' accs
    for mut_no in range(1, total_mutant_num+1): # mut_no starts from 1
        mutant_accs = []
        for model_no in range(tatal_model_training):
            mutants_path = f'{DMPP_HOME_PATH}/mutated_models_{subject_name}_trained:{model_no}/{op_no}'
            for path in sorted(os.listdir(mutants_path)):
                if path.endswith(f"_{mut_no}.h5"):
                    mutant_prediction_info = get_prediction_array_dmpp(base_model, path[:-3], mutants_path)
                    mutant_acc = len(np.argwhere(mutant_prediction_info == True)) / len(mutant_prediction_info) * 100                    
                    mutant_accs.append(mutant_acc)
                    break

        print(f'Original accs: {orig_accs}')
        print(f'Mutant {mut_no} accs: {mutant_accs}')        
        is_sts, p_value, effect_size = is_diff_sts_classification(orig_accs, mutant_accs)
        with open(f'./output/mutants_accs.csv', 'a') as f:
            for model_no in range(tatal_model_training):
                f.write(f'{op_no},{mut_no},{model_no},{mutant_accs[model_no]:.3f},{is_sts}\n')

        # Check killed or not        
        if is_sts:
            print(f'Mutant {mut_no} is STS with p-value {p_value} and effect size {effect_size}')
        else:
            print(f'Mutant {mut_no} is not STS')
            # Remove the mutants if not killed (due to space limit of my computer)
            for model_no in range(tatal_model_training):
                mutants_path = f'{DMPP_HOME_PATH}/mutated_models_{subject_name}_trained:{model_no}/{op_no}'
                for path in sorted(os.listdir(mutants_path)):
                    if path.endswith(f"_{mut_no}.h5"):
                        os.remove(os.path.join(mutants_path, path))
                        break
