import numpy as np
import os, sys
from deepcrime_scripts.utils import *


DMPP_HOME_PATH = '/home/DC_replication/DCReplication/deepmutationpp'


def get_prediction_array(base_model, name_prefix, model_dir, num_models): 
    prediction_info = []    
    for i in range(num_models):        
        model_file = os.path.join(model_dir, f"{name_prefix}_{i}.h5")
        prediction_info.append(base_model.get_prediction_info(model_file))

    return prediction_info


def get_prediction_array_mutant(base_model, mutant_paths): 
    # This function just takes pathes of mutants; functionality is same as get_prediction_array
    prediction_info = []    
    for mutant_path in mutant_paths:               
        prediction_info.append(base_model.get_prediction_info(mutant_path))
    return prediction_info


def save_killing_info(original_prediction_info, mutation_prediction_info, output_dir, mutation_prefix):
    killing_info = []

    for i in range(0, len(original_prediction_info)):
        killing_array = np.empty(len(original_prediction_info[i]))
        for j in range(0, len(original_prediction_info[i])):                        
            if original_prediction_info[i][j] == True and mutation_prediction_info[i][j] == False:
                killing_array[j] = 1
            else:
                killing_array[j] = 0
        killing_info.append(killing_array)

    killing_info = np.asarray(killing_info)
    np.save(os.path.join(output_dir, mutation_prefix + '_ki.npy'), killing_info)


if __name__ == "__main__":                
    subject_name = 'audio'        
    num_models = 20
    num_mutants = 20
    # ops = [0, 1, 2, 3, 4, 7] #, 5, 6 not applicable
    ops = [sys.argv[1]]
    orig_model_dir = f'{DMPP_HOME_PATH}/original_models/multiple_training/'
    
    base_model = get_subject_model(subject_name)
    orig_prediction_info = get_prediction_array(base_model, f'{subject_name}_original', orig_model_dir, num_models)
    for op_no in ops:
        for mut_no in range(1, num_mutants+1):
            mutant_paths = []
            for model_no in range(num_models):                            
                mutants_dir = f'{DMPP_HOME_PATH}/mutated_models_{subject_name}_trained:{model_no}/{op_no}'
                for path in sorted(os.listdir(mutants_dir)):
                    if path.endswith(f"_{mut_no}.h5"):                            
                        mutant_paths.append(mutants_dir + '/' + path)
                        break
            if len(mutant_paths) != num_models:
                continue

            mutant_prediction_info = get_prediction_array_mutant(base_model, mutant_paths)

            output_dir = './output/ki'
            prefix = f'{subject_name}_op:{op_no}_mutant:{mut_no}'

            print(prefix)
            print(len(mutant_prediction_info), len(mutant_prediction_info[0]))
            print(orig_prediction_info[0])
            print(mutant_prediction_info[0])

            print(orig_prediction_info[0][0])
            print(type(orig_prediction_info[0][0]))
            print(orig_prediction_info[0][0] is True)
            print(orig_prediction_info[0][0] == True)

            save_killing_info(orig_prediction_info, mutant_prediction_info, output_dir, prefix)            
