import tensorflow as tf

from utils import helpers
from utils.helpers import one_hot_it


def get_labels_from_annotation(annotation_tensor, class_labels):
    """Returns tensor of size (width, height, num_classes) derived from annotation tensor.
    The function returns tensor that is of a size (width, height, num_classes) which
    is derived from annotation tensor with sizes (width, height) where value at
    each position represents a class. The functions requires a list with class
    values like [0, 1, 2 ,3] -- they are used to derive labels. Derived values will
    be ordered in the same way as the class numbers were provided in the list. Last
    value in the aforementioned list represents a value that indicate that the pixel
    should be masked out. So, the size of num_classes := len(class_labels) - 1.

    Parameters
    ----------
    annotation_tensor : Tensor of size (width, height)
        Tensor with class labels for each element
    class_labels : list of ints
        List that contains the numbers that represent classes. Last
        value in the list should represent the number that was used
        for masking out.

    Returns
    -------
    labels_2d_stacked : Tensor of size (width, height, num_classes).
        Tensor with labels for each pixel.
    """

    # Last value in the classes list should show
    # which number was used in the annotation to mask out
    # the ambigious regions or regions that should not be
    # used for training.
    # TODO: probably replace class_labels list with some custom object
    valid_entries_class_labels = list(class_labels)[:-1]

    # Stack the binary masks for each class
    labels_2d = [tf.equal(annotation_tensor, x) for x in valid_entries_class_labels]

    # Perform the merging of all of the binary masks into one matrix
    labels_2d_stacked = tf.stack(labels_2d, axis=2)

    # Convert tf.bool to tf.float
    # Later on in the labels and logits will be used
    # in tf.softmax_cross_entropy_with_logits() function
    # where they have to be of the float type.
    labels_2d_stacked_float = tf.to_float(labels_2d_stacked)

    return labels_2d_stacked_float


def get_labels_from_annotation_batch(annotation_batch_tensor, class_labels):
    """
    Returns
    -------
    batch_labels : Tensor of size (batch_size, width, height, num_classes).
        Tensor with labels for each batch.
    """

    batch_labels = tf.map_fn(fn=lambda x: one_hot_it(label=x, label_values=class_labels),
                             elems=annotation_batch_tensor,
                             dtype=tf.float32)

    return batch_labels


def get_valid_entries_indices_from_annotation_batch(annotation_batch_tensor):
    mask_out_class_label = 255.0

    valid_labels_mask = tf.not_equal(annotation_batch_tensor,
                                     mask_out_class_label)

    valid_labels_indices = tf.where(valid_labels_mask)

    return tf.to_int32(valid_labels_indices)


def get_valid_logits_and_labels(labels_batch, logits_batch):
    raw_prediction = tf.reshape(logits_batch, [-1, 3])
    gt = tf.reshape(labels_batch, [-1])
    # supposed 2 is the ignored label
    indices = tf.squeeze(tf.where(tf.not_equal(gt, 255)), 1)
    gt = tf.cast(tf.gather(gt, indices), tf.int32)
    prediction = tf.gather(raw_prediction, indices)

    return gt, prediction