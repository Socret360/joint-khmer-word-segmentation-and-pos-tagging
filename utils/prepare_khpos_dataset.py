import os
import argparse
from str2bool import str2bool
from read_pos_map import read_pos_map
from read_char_map import read_char_map

parser = argparse.ArgumentParser(description='Prepare khPOS dataset to correct format.')
parser.add_argument('data_dir', type=str, help='Path to dataset directory.')
parser.add_argument('--split_sentences', type=str2bool, nargs='?',
                    help='whether to generate more samples by splitting the sentences into chunks of 0 to length.', default=False)
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


def write_samples_to_file(input_filepath, output_filepath, split_sentences=False):
    writer = open(output_filepath, "w")
    count = 0
    MAX_SENTENCE_LENGTH = -999
    with open(input_filepath, "r") as data_file:
        lines = data_file.readlines()
        for i, line in enumerate(lines):
            if split_sentences:
                chunks = line.strip("\n").split(" ")
                for start in range(0, len(chunks)+1):
                    for end in range(start+1, len(chunks)+1):
                        sub_line = " ".join(chunks[start:end])
                        X, y = process_line(sub_line)
                        writer.write(X + "\t" + y + "\n")
                        count += 1

            X, y = process_line(line)
            if MAX_SENTENCE_LENGTH < len(X):
                MAX_SENTENCE_LENGTH = len(X)
            writer.write(X + "\t" + y + ("\n" if i != len(lines)-1 else ""))
            count += 1

    writer.close()
    print(f"written samples: {count} | MAX_SENTENCE_LENGTH: {MAX_SENTENCE_LENGTH}")


if __name__ == "__main__":
    training_file = os.path.join(args.data_dir, "corpus-draft-ver-1.0/data/after-replace/train.all2")
    test_file_open = os.path.join(args.data_dir, "corpus-draft-ver-1.0/data/OPEN-TEST")
    test_file_close = os.path.join(args.data_dir, "corpus-draft-ver-1.0/data/CLOSE-TEST")

    write_samples_to_file(training_file, os.path.join(args.output_dir, "train.txt"), args.split_sentences)
    write_samples_to_file(test_file_open, os.path.join(args.output_dir, "test_open.txt"))
    write_samples_to_file(test_file_close, os.path.join(args.output_dir, "test_close.txt"))
