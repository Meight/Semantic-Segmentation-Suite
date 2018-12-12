from utils.naming.naming import FilesFormatterFactory
import numpy as np

from utils.utils import gather_multi_label_data

dataset_name = 'a'
model_name = 'b'
backbone_name = 'c'
training_parameters = {
    'd': 1
}

paths = gather_multi_label_data('test_dataset')

print(paths)