import sys
import os

class FilesFormatterFactory:
    def __init__(self, mode, dataset_name, model_name, backbone_name, training_parameters, results_folder='results',
                 verbose=False):
        self.mode = mode
        self.training_parameters = training_parameters
        self.backbone_name = backbone_name
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.verbose = verbose

        self.results_folder = results_folder
        self._full_detailed_path = self.generate_full_detailed_path()
        self._parameters_string = self.generate_parameters_string()

        if not sys.path.exists(self._full_detailed_path):
            sys.path.mkdirs(self._full_detailed_path)

    def generate_full_detailed_path(self):
        return sys.path.join(self.results_folder,
                             self.mode,
                             self.dataset_name,
                             self.model_name,
                             self.backbone_name,
                             self._parameters_string)

    def generate_parameters_string(self):
        return '_'.join(['{}-{}'.format(self._get_initials(parameter_name), parameter_value)
                         for parameter_name, parameter_value in self.training_parameters.items()])

    def _get_initials(self, string):
        return ''.join([x[0].upper() for x in string.split('_')])

    def generate_checkpoint_name(self, current_epoch):
        return sys.path.join(self._full_detailed_path, current_epoch + '.ckpt')

    def generate_summary_name(self, current_epoch):
        return sys.path.join(self._full_detailed_path, current_epoch + '.csv')

    def get_checkpoint_formatter(self, saver):
        return CheckpointFormatter(self.mode,
                                   self.dataset_name,
                                   self.model_name,
                                   self.backbone_name,
                                   self.training_parameters,
                                   saver,
                                   self.verbose)

    def get_summary_formatter(self):
        return SummaryFormatter(self.mode,
                                self.dataset_name,
                                self.model_name,
                                self.backbone_name,
                                self.training_parameters,
                                self.verbose)


class SummaryFormatter(FilesFormatterFactory):
    def __init__(self, mode, dataset_name, model_name, backbone_name, training_parameters, verbose=False):
        super().__init__(mode, dataset_name, model_name, backbone_name, training_parameters, verbose=verbose)
        self.header_created = False
        self.current_epoch = 0

    def update(self, current_epoch, measures_dictionary, column_margin=2, precision=3, verbose=False):
        self.current_epoch = current_epoch
        column_width = len(max(measures_dictionary.keys(), key=len)) + column_margin

        with open(self.generate_summary_name(current_epoch=self.current_epoch), 'a+') as summary_file:
            if not self.header_created:
                summary_file.write(self._generate_header(column_names=measures_dictionary.keys(),
                                                         column_width=column_width))
                self.header_created = True

            summary_file.write(''.join(['{value:<{width}.{precision}f}'.format(value=measure,
                                                                               width=column_width,
                                                                               precision=precision)
                                        for measure in measures_dictionary.values()]))

        if self.verbose or verbose:
            print('Updated session summary at {}.'.format(self.generate_summary_name(self.current_epoch)))

    def _generate_header(self, column_names, column_width):
        return ''.join(['{0:<{width}}'.format(column_name, width=column_width)
                        for column_name in column_names]) + '\n'


class CheckpointFormatter(FilesFormatterFactory):
    def __init__(self, mode, dataset_name, model_name, backbone_name, training_parameters, saver, verbose=False):
        super().__init__(mode, dataset_name, model_name, backbone_name, training_parameters, verbose=verbose)
        self.saver = saver
        self.current_epoch = 0

    def save(self, session, current_epoch, verbose=False):
        self.current_epoch = current_epoch
        self.saver.save(session, self.generate_checkpoint_name(current_epoch=self.current_epoch))

        if self.verbose or verbose:
            print('Saved checkpoints for epoch {} at {}.'.format(self.current_epoch,
                                                                 self.generate_checkpoint_name(
                                                                     current_epoch=self.current_epoch)))

    def restore(self, session, model_checkpoint_name):
        self.saver.restore(session, model_checkpoint_name)