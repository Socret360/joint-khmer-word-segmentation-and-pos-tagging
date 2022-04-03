import os
import argparse
from re import S
import numpy as np
import tensorflow as tf
from str2bool import str2bool
from read_pos_map import read_pos_map
from read_char_map import read_char_map

parser = argparse.ArgumentParser(description='Prepare khPOS dataset to correct format.')
parser.add_argument('data_dir', type=str, help='Path to dataset directory.')
parser.add_argument('char_map', type=str, help='path to characters map file.')
parser.add_argument('pos_map', type=str, help='path to pos map file.')
parser.add_argument('--split_sentences', type=str2bool, nargs='?',
                    help='whether to generate more samples by splitting the sentences into chunks of 0 to length.', default=False)
parser.add_argument('--output_dir', type=str,
                    help='The directory to output the results.', default="output")
args = parser.parse_args()

assert os.path.exists(args.data_dir), "data_dir does not exist"
os.makedirs(args.output_dir, exist_ok=True)

pos_map, pos_to_index, _ = read_pos_map(args.pos_map)
char_map, char_to_index, _ = read_char_map(args.char_map)

print(char_to_index)

REVISED_TAG = {
    "M": "NN",
    "RPN": "PRO",
    "CUR": "SYM",
    "DBL": "SYM",
    "ETC": "SYM",
    "KAN": "SYM",
    "UH": "PA",
    "VB_JJ": "VB",
    "VCOM": "VB"
}


def process_line(line):
    X = ""
    y = ""
    chunks = line.strip("\n").split(" ")
    for chunk in chunks:
        word, tag = chunk.split("/")
        if tag in REVISED_TAG:
            tag = REVISED_TAG[tag]

        X += word

        y += f"/{tag}"

        for _ in word[1:]:
            y += "/NS"

    return X, y


def write_samples_to_tfrecord(input_filepath, output_filepath, split_sentences=False):
    writer = tf.io.TFRecordWriter(output_filepath)
    count = 0
    MAX_SENTENCE_LENGTH = -999
    with open(input_filepath, "r") as data_file:
        lines = data_file.readlines()
        for _, line in enumerate(lines):
            if split_sentences:
                chunks = line.strip("\n").split(" ")
                for start in range(0, len(chunks)+1):
                    for end in range(start+1, len(chunks)+1):
                        sub_line = " ".join(chunks[start:end])
                        X, y = process_line(sub_line)
                        out = prepare_single_line(X, y)
                        writer.write(out)
                        count += 1

            X, y = process_line(line)
            if MAX_SENTENCE_LENGTH < len(X):
                MAX_SENTENCE_LENGTH = len(X)
            out = prepare_single_line(X, y)
            writer.write(out)
            count += 1

    writer.close()
    print(f"written samples: {count} | MAX_SENTENCE_LENGTH: {MAX_SENTENCE_LENGTH}")


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


def prepare_single_line(sentence, sentence_tag):
    s, t = [], []
    for char in sentence:
        if char in char_to_index:
            char_index = char_to_index[char]
        else:
            char_index = char_to_index["UNK"]
        s.append(char_index)

    for pos in sentence_tag.split("/")[1:]:
        pos_index = pos_to_index[pos]
        t.append(pos_index)

    data = {
        'sentence': tf.train.Feature(int64_list=tf.train.Int64List(value=s)),
        'sentence_tag': tf.train.Feature(int64_list=tf.train.Int64List(value=t)),
    }
    out = tf.train.Example(features=tf.train.Features(feature=data))
    return out.SerializeToString()


if __name__ == "__main__":
    training_file = os.path.join(args.data_dir, "corpus-draft-ver-1.0/data/after-replace/train.all2")
    test_file_open = os.path.join(args.data_dir, "corpus-draft-ver-1.0/data/OPEN-TEST")
    test_file_close = os.path.join(args.data_dir, "corpus-draft-ver-1.0/data/CLOSE-TEST")

    write_samples_to_tfrecord(training_file, os.path.join(args.output_dir, "train.tfrecord"), args.split_sentences)
    write_samples_to_tfrecord(test_file_open, os.path.join(args.output_dir, "test_open.tfrecord"))
    write_samples_to_tfrecord(test_file_close, os.path.join(args.output_dir, "test_close.tfrecord"))
