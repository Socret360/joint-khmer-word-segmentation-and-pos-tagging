from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import os
import json
import numpy as np
import argparse
from network import Network
from utils import str2bool, read_pos_map, read_char_map, read_config, parse_tf_record_element

parser = argparse.ArgumentParser(description='Start the training process.')
parser.add_argument('config', type=str, help='path to config file.')
parser.add_argument('train_set', type=str, help='path to training dataset.')
parser.add_argument('char_map', type=str, help='path to characters map file.')
parser.add_argument('pos_map', type=str, help='path to pos map file.')
parser.add_argument('--shuffle', type=str2bool, nargs='?',
                    help='whether to shuffle the dataset when creating the batch', default=True)
parser.add_argument('--epochs', type=int,
                    help='the number of epochs to train', default=100)
parser.add_argument('--output_dir', type=str,
                    help='path to output directory.', default="output")
args = parser.parse_args()


assert args.epochs > 0, "epochs must be larger than zero"

config = read_config(args.config)

os.makedirs(args.output_dir, exist_ok=True)

char_map, char_to_index, index_to_char = read_char_map(args.char_map)
pos_map, pos_to_index, index_to_pos = read_pos_map(args.pos_map)

model = Network(
    output_dim=len(pos_map),
    embedding_dim=len(char_map),
    num_stacks=config["model"]["num_stacks"],
    hidden_layers_dim=config["model"]["hidden_layers_dim"],
    max_sentence_length=config["model"]["max_sentence_length"],
)

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=config["training"]["learning_rate"])
)


def count_tfrecord_examples(
        tfrecords_dir: str,
) -> int:
    """
    Counts the total number of examples in a collection of TFRecord files.

    :param tfrecords_dir: directory that is assumed to contain only TFRecord files
    :return: the total number of examples in the collection of TFRecord files
        found in the specified directory
    """

    count = 0
    for file_name in os.listdir(tfrecords_dir):
        tfrecord_path = os.path.join(tfrecords_dir, file_name)
        count += sum(1 for _ in tf.data.TFRecordDataset(tfrecord_path))

    return count


dataset = tf.data.TFRecordDataset(args.train_set)
dataset = dataset.map(lambda x: parse_tf_record_element(x, len(char_map), len(pos_map), config["model"]["max_sentence_length"]))
num_samples = sum(1 for _ in dataset)

dataset = dataset.batch(config["training"]["batch_size"])

print(num_samples)

model.fit(
    x=dataset,
    shuffle=args.shuffle,
    epochs=args.epochs,
    batch_size=config["training"]["batch_size"],
    steps_per_epoch=num_samples//config["training"]["batch_size"],
    callbacks=[
        ModelCheckpoint(
            filepath=os.path.join(
                args.output_dir,
                "cp_{epoch:02d}_loss-{loss:.2f}.h5"
            ),
            save_weights_only=False,
            save_best_only=True,
            monitor='loss',
            mode='min'
        ),
    ]
)

model.save_weights(os.path.join(args.output_dir, "model.h5"))
