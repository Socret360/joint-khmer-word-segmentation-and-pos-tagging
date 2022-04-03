from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from gc import callbacks
import os
import json
import argparse
from network import Network
from data_generator import DataGenerator
from utils import str2bool, read_pos_map, read_char_map, read_config, read_samples

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

char_map = read_char_map(args.char_map)
pos_map = read_pos_map(args.pos_map)
samples = read_samples(args.train_set)

data_generator = DataGenerator(
    samples=samples,
    pos_map=pos_map,
    char_map=char_map,
    shuffle=args.shuffle,
    batch_size=config["training"]["batch_size"],
    max_sentence_length=config["model"]["max_sentence_length"],
)

if args.colab_tpu:
    # Get a handle to the attached TPU. On GCP it will be the CloudTPU itself
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
    # Connect to the TPU handle and initialise it
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

    with strategy.scope():
        model = Network(
            output_dim=len(pos_map),
            embedding_dim=len(char_map),
            num_stacks=config["model"]["num_stacks"],
            batch_size=config["training"]["batch_size"] * strategy.num_replicas_in_sync,
            hidden_layers_dim=config["model"]["hidden_layers_dim"],
            max_sentence_length=config["model"]["max_sentence_length"],
        )
        model.summary()

        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=config["training"]["learning_rate"])
        )

        model.fit(
            x=tf.data.Dataset.from_generator(data_generator),
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

else:
    model = Network(
        output_dim=len(pos_map),
        embedding_dim=len(char_map),
        num_stacks=config["model"]["num_stacks"],
        batch_size=config["training"]["batch_size"],
        hidden_layers_dim=config["model"]["hidden_layers_dim"],
        max_sentence_length=config["model"]["max_sentence_length"],
    )

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=config["training"]["learning_rate"])
    )

    model.fit(
        x=data_generator,
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
