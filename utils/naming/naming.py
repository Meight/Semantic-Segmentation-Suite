import sys

class FilesFormatter:
    def __init__(self, mode, dataset_name, model_name, backbone_name, training_parameters, results_folder='results'):
        self.mode = mode
        self.training_parameters = training_parameters
        self.backbone_name = backbone_name
        self.model_name = model_name
        self.dataset_name = dataset_name

        self.results_folder = results_folder
        self._full_detailed_path = self.generate_full_detailed_path()
        self._parameters_string = self.generate_parameters_string()

    def generate_full_detailed_path(self):
        return sys.path.join(self.results_folder,
                             self.mode,
                             self.dataset_name,
                             self.model_name,
                             self.backbone_name)

    def generate_parameters_string(self):
        return '_'.join(['{}-{}'.format(self._get_initials(parameter_name), parameter_value)
                         for parameter_name, parameter_value in self.training_parameters.items()])

    def _get_initials(self, string):
        return ''.join([x[0].upper() for x in string.split('_')])

    def generate_checkpoint_name(self, current_epoch):
        return sys.path.join(self._full_detailed_path, self._parameters_string, current_epoch + '.ckpt')

    def generate_summary_name(self, current_epoch):
        return sys.path.join(self._full_detailed_path, self._parameters_string, current_epoch + '.csv')


class SummaryFormatter(FilesFormatter):
    def __init__(self):
        self.header_created = False

    def __enter__(self, current_epoch, measures_dictionary, column_margin=2):
        column_width = len(max(measures_dictionary.keys(), key=len)) + column_margin

        with open(self.generate_summary_name(current_epoch=current_epoch), 'a') as summary_file:
            if not self.header_created:
                summary_file.write(self.generate_header(column_names=measures_dictionary.keys(),
                                                        column_width=column_width))

            summary_file.write(''.join(['{value:<{width}.3f}'.format(value=measure, width=column_width)
                                       for measure in measures_dictionary.values()]))

    def generate_header(self, column_names, column_width):
        return ''.join(['{0:<{width}}'.format(column_name, width=column_width)
                          for column_name in column_names]) + '\n'