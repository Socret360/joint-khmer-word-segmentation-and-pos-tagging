import os
import shutil
import argparse
import numpy as np
import tensorflow as tf
from network import Network
from utils import read_config, read_char_map, read_pos_map

SUPPORTED_TYPES = [
    "keras",
    "tflite"
]

parser = argparse.ArgumentParser(
    description='Converts a pretrained weights into keras or tflite format.')
parser.add_argument('config', type=str, help='path to config file.')
parser.add_argument('weights', type=str, help='path to the weight file.')
parser.add_argument('char_map', type=str, help='path to characters map file.')
parser.add_argument('pos_map', type=str, help='path to pos map file.')
parser.add_argument('--output_dir', type=str,
                    help='path to output directory.', default="output")
parser.add_argument('--output_type', type=str,
                    help='the type of the output model. One of type: "keras", "tflite"', default="keras")
args = parser.parse_args()


assert args.output_type in SUPPORTED_TYPES, f"{args.output_type} is not supported yet. Please choose one of type {SUPPORTED_TYPES}"

config = read_config(args.config)

os.makedirs(args.output_dir, exist_ok=True)

char_map = read_char_map(args.char_map)
pos_map = read_pos_map(args.pos_map)

model = Network(
    output_dim=len(pos_map),
    embedding_dim=len(char_map),
    num_stacks=config["model"]["num_stacks"],
    hidden_layers_dim=config["model"]["hidden_layers_dim"],
    max_sentence_length=config["model"]["max_sentence_length"],
)

model.summary()

model.load_weights(args.weights)

config_file_name = os.path.basename(args.config)
config_file_name = config_file_name[:config_file_name.index(".")]

if args.output_type == "keras":
    model.save(os.path.join(args.output_dir, f"{config_file_name}.h5"))
elif args.output_type == "tflite":
    temp_saved_model = os.path.join(args.output_dir, f"{config_file_name}")
    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, config["model"]["max_sentence_length"], len(char_map)], model.inputs[0].dtype))
    model.save(temp_saved_model, save_format="tf", signatures=concrete_func)

    tflite_converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir=temp_saved_model, signature_keys=['serving_default'])
    tflite_model = tflite_converter.convert()
    output_file_path = os.path.join(args.output_dir, f"{config_file_name}.tflite")
    open(output_file_path, 'wb').write(tflite_model)

    shutil.rmtree(temp_saved_model)
