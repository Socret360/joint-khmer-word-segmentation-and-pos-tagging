import tensorflow as tf


def parse_tf_record_element(element, num_chars, num_pos, char_to_index, pos_to_index, max_sentence_length):
    data = {
        'sentence': tf.io.FixedLenFeature([], tf.string),
        'sentence_tag': tf.io.FixedLenFeature([], tf.string),
    }
    content = tf.io.parse_single_example(element, data)
    sentence_indices = tf.io.parse_tensor(content['sentence'], out_type=tf.int32)
    sentence_tag_indices = tf.io.parse_tensor(content['sentence_tag'], out_type=tf.int32)
    num_paddings = max_sentence_length - tf.shape(sentence_indices)[0]

    sentence_input_vector = tf.concat([
        tf.one_hot(sentence_indices, num_chars, on_value=1.0, off_value=0.0),
        tf.zeros([num_paddings, num_chars]),
    ], axis=0)
    sentence_output_vector = tf.concat([
        tf.one_hot(sentence_tag_indices, num_pos, on_value=1.0, off_value=0.0),
        tf.zeros([num_paddings, num_pos]),
    ], axis=0)

    return (sentence_input_vector, sentence_output_vector)
