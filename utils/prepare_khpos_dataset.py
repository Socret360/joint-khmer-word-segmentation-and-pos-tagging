import os
import argparse
from str2bool import str2bool

parser = argparse.ArgumentParser(description='Prepare khPOS dataset to correct format.')
parser.add_argument('data_dir', type=str, help='Path to dataset directory.')
parser.add_argument('--word_based', type=str2bool, nargs='?',
                    help='whether to convert the dataset from sentence based to word based.', default=False)
parser.add_argument('--output_dir', type=str,
                    help='The directory to output the results.', default="output")
args = parser.parse_args()

assert os.path.exists(args.data_dir), "data_dir does not exist"
os.makedirs(args.output_dir, exist_ok=True)


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


def process_dataset(filepath, word_based=False):
    samples = []
    with open(filepath, "r") as data_file:
        lines = data_file.readlines()
        for i, line in enumerate(lines):
            if word_based:
                chunks = line.strip("\n").split(" ")
                for j, chunk in enumerate(chunks):
                    X, y = process_line(chunk)
                    samples.append(f"{X}\t{y}")
                    if i != len(lines) - 1 or j != len(chunks):
                        samples.append("\n")

            X, y = process_line(line)
            samples.append(f"{X}\t{y}")
            if i != len(lines) - 1:
                samples.append("\n")

    return samples


def write_samples_to_file(samples, output_filepath):
    with open(output_filepath, "w") as output_file:
        output_file.writelines(samples)


if __name__ == "__main__":
    training_file = os.path.join(args.data_dir, "corpus-draft-ver-1.0/data/after-replace/train.all2")
    test_file_open = os.path.join(args.data_dir, "corpus-draft-ver-1.0/data/OPEN-TEST")
    test_file_close = os.path.join(args.data_dir, "corpus-draft-ver-1.0/data/CLOSE-TEST")

    write_samples_to_file(process_dataset(training_file, args.word_based), os.path.join(args.output_dir, "train.txt"))
    write_samples_to_file(process_dataset(test_file_open), os.path.join(args.output_dir, "test_open.txt"))
    write_samples_to_file(process_dataset(test_file_close), os.path.join(args.output_dir, "test_close.txt"))
