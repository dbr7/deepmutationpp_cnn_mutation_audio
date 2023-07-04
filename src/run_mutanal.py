import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import datetime

from deepmutationpp.cnn_mutation_audio.src.data_sort import check_data
from deepmutationpp.cnn_mutation_audio.src.audio_model import get_test_df

print(datetime.datetime.now())

model_path = os.path.join('..', '..', 'original_models', 'audio_original_0.h5')
mutants_path = os.path.join('..', '..', 'mutated_models_audio')
results_path = os.path.join('..', '..', 'results_audio')

dataset_test = get_test_df('T')

check_data(model_path, mutants_path, dataset_test, results_path, "test")

dataset_weak = get_test_df('W')

check_data(model_path, mutants_path, dataset_weak, results_path, "weak")


print(datetime.datetime.now())