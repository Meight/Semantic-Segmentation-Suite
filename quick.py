from utils.naming.naming import FilesFormatterFactory
import numpy as np
dataset_name = 'a'
model_name = 'b'
backbone_name = 'c'
training_parameters = {
    'd': 1
}

files_formatter_factory = FilesFormatterFactory(mode='training',
                                                dataset_name=dataset_name,
                                                model_name=model_name,
                                                backbone_name=backbone_name,
                                                training_parameters=training_parameters,
                                                verbose=True,
                                                results_folder='C:/Work/Python/Semantic-Segmentation-Suite/')


summary_formatter = files_formatter_factory.get_summary_formatter()
avg_score = np.mean([5, 10])
class_avg_scores = 23
avg_precision = 12

measures = {
    'accuracy': avg_score,
    'class_accuracies': class_avg_scores,
    'precision': avg_precision,
}

summary_formatter.update(1, measures)