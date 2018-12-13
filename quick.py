from pprint import pprint
import matplotlib.pyplot as plt

from utils import helpers
from utils.naming.naming import FilesFormatterFactory
import numpy as np

from utils.utils import gather_multi_label_data, to_n_hot_encoded, get_all_available_annotation_resized_tensors_for_image, \
    load_image

class_names_list, label_values = helpers.get_label_info("./test_dataset/class_dict.csv")

class_colors_dictionary = dict(zip(class_names_list, label_values))

pprint(class_colors_dictionary)

paths = gather_multi_label_data('test_dataset')

for image_path, image_masks in paths['train'].items():
    print(image_path)

    input_height, input_width = 512, 512
    n_encoded_masks = get_all_available_annotation_resized_tensors_for_image((input_height, input_width), image_masks, class_colors_dictionary)

    if not n_encoded_masks:
        continue
    tensor = n_encoded_masks[0]

    print(np.shape(tensor))

    plt.figure()
    for k in range(21):
        plt.subplot(5, 5, k + 1)
        plt.imshow(tensor[:, :, k])
        plt.title(class_names_list[k])

    plt.show()