from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from gc import callbacks
import os
import json
import argparse
from network import Network
from data_generator import DataGenerator
from utils import str2bool, read_pos_map, read_char_map, read_config, parse_tf_record_element

parser = argparse.ArgumentParser(description='Start the training process.')
parser.add_argument('config', type=str, help='path to config file.')
parser.add_argument('train_set', type=str, help='path to training dataset.')
parser.add_argument('char_map', type=str, help='path to characters map file.')
parser.add_argument('pos_map', type=str, help='path to pos map file.')
parser.add_argument('--shuffle', type=str2bool, nargs='?',
                    help='whether to shuffle the dataset when creating the batch', default=True)
parser.add_argument('--colab_tpu', type=str2bool, nargs='?',
                    help='whether to use google colab\'s tpu', default=False)
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

if args.colab_tpu:
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
        print("Running on TPU ", resolver.master())
        print("REPLICAS: ", strategy.num_replicas_in_sync)
    except:
        print("WARNING: No TPU detected.")
        strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        batch_size = config["training"]["batch_size"] * strategy.num_replicas_in_sync
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

        dataset = tf.data.TFRecordDataset(args.train_set)
        dataset = dataset.map(lambda x: parse_tf_record_element(x, len(char_map), len(pos_map), config["model"]["max_sentence_length"]))
        dataset = dataset.batch(batch_size)

        model.fit(
            dataset,
            shuffle=args.shuffle,
            epochs=args.epochs,
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

else:
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

    dataset = tf.data.TFRecordDataset(args.train_set)
    dataset = dataset.map(lambda x: parse_tf_record_element(x, len(char_map), len(pos_map), char_to_index, pos_to_index, config["model"]["max_sentence_length"]))

    model.fit(
        x=dataset,
        shuffle=args.shuffle,
        epochs=args.epochs,
        batch_size=config["training"]["batch_size"],
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
