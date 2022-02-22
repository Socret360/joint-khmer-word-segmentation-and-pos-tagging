import os
import json
import argparse
import numpy as np
from network import Network
from utils import read_pos_map, read_char_map, read_config, read_samples

parser = argparse.ArgumentParser(description='Start the test process.')
parser.add_argument('config', type=str, help='path to config file.')
parser.add_argument('test_set', type=str, help='path to test dataset.')
parser.add_argument('char_map', type=str, help='path to characters map file.')
parser.add_argument('pos_map', type=str, help='path to pos map file.')
parser.add_argument('weights', type=str, help='path to weights file.')
args = parser.parse_args()

assert os.path.exists(args.weights), "weights doest not exist."

config = read_config(args.config)
char_map = read_char_map(args.char_map)
pos_map = read_pos_map(args.pos_map)
samples = read_samples(args.test_set)

num_chars = len(char_map)
num_pos = len(pos_map)
pos_to_index = {pos: i for i, pos in enumerate(pos_map)}
index_to_pos = {i: pos for i, pos in enumerate(pos_map)}
char_to_index = {char: i for i, char in enumerate(char_map)}
index_to_char = {i: char for i, char in enumerate(char_map)}

model = Network(
    output_dim=len(pos_map),
    embedding_dim=len(char_map),
    num_stacks=config["model"]["num_stacks"],
    batch_size=1,
    hidden_layers_dim=config["model"]["hidden_layers_dim"],
)

model.load_weights(args.weights, by_name=True)

pos_count = {pos: {"correct": 0, "corpus": 0} for pos in pos_map}

with open("output/test.txt", "w") as output_file:
    for sample in samples:
        sentence, sentence_tag = sample.split("\t")
        sentence_input_vector = np.zeros((len(sentence), num_chars))
        sentence_output_vector = np.zeros((len(sentence), num_pos))
        for i, char in enumerate(sentence):
            if char in char_to_index:
                char_index = char_to_index[char]
            else:
                char_index = char_to_index["UNK"]
            sentence_input_vector[i, char_index] = 1

        for i, pos in enumerate(sentence_tag.split("/")[1:]):
            pos_index = pos_to_index[pos]
            sentence_output_vector[i, pos_index] = 1
            pos_count[index_to_pos[pos_index]]["corpus"] += 1

        pred = model.predict(np.array([sentence_input_vector]))[0]

        for char_idx, pos_vector in enumerate(pred):
            pos_index_pred = np.argmax(pos_vector)
            pos_index_target = np.argmax(sentence_output_vector[char_idx])

            if pos_index_pred == pos_index_target:
                pos_count[index_to_pos[pos_index_pred]]["correct"] += 1

            # words, tmp = [], []
            # for char_idx, pos_vector in enumerate(pred):
            #     pos_index = np.argmax(pos_vector)
            #     if index_to_pos[pos_index] == "NS":
            #         tmp.append(sentence[char_idx])
            #     else:
            #         if len(tmp) > 0:
            #             words.append("".join(tmp))
            #             tmp = []
            #         tmp.append(sentence[char_idx])
            # if len(tmp) > 0:
            #     words.append("".join(tmp))

            # output_file.write(f"{sentence}\t{' '.join(words)}\n")

    for pos in pos_count:
        if pos not in ["NS"]:
            accuracy = round((pos_count[pos]["correct"] / pos_count[pos]["corpus"])*100, 2)
            print(f"-- {pos}: {accuracy}")
