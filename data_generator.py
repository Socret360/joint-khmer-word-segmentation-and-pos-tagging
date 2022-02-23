import numpy as np
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    """ A generator used for creating batches of samples during training.

    Args
    ---
    - samples: A python list where each item is a string of sentence and sentence tag separated by TAB character.
    - char_map: A python list where each item is a string representing a character.
    - pos_map: A python list where each item is a string representing a tag.
    - batch_size: An int representing the number of samples produce on each batch. (Defaults to 128)
    - shuffle: A boolean on whether to shuffle each samples in a batch.

    Returns
    ---
    A Keras Data Generator.

    Raises
    ---
    - `samples` must not be empty.
    - elements in `samples` must be larger than batch size.
    - `char_map` must not be empty.
    - `pos_map` must not be empty.

    References
    ---
    - Buoy, R., Taing, N., & Kor, S. (2021). Joint Khmer Word Segmentation and Part-of-Speech Tagging Using Deep Learning. arXiv preprint arXiv:2103.16801.
    - Loem, M. (2021, May 4). Joint Khmer Word Segmentation and POS tagging. Medium. Retrieved February 22, 2022, from https://towardsdatascience.com/joint-khmer-word-segmentation-and-pos-tagging-cad650e78d30 
    """

    def __init__(self, samples, char_map, pos_map, batch_size=128, shuffle=False):
        assert len(samples) > 0, "samples must not be empty."
        assert len(samples) > batch_size, "the number of samples must be larger than batch size."
        assert len(char_map) > 0, "char_map must not be empty."
        assert len(pos_map) > 0, "pos_map must not be empty."

        self.samples = samples
        self.shuffle = shuffle
        self.num_pos = len(pos_map)
        self.batch_size = batch_size
        self.num_chars = len(char_map)
        self.indices = range(0, len(self.samples))
        self.pos_to_index = {pos: i for i, pos in enumerate(pos_map)}
        self.char_to_index = {char: i for i, char in enumerate(char_map)}
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]
        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        X = []
        y = []

        max_sentence_length = max([len(self.samples[sample_idx].split("\t")[0]) for sample_idx in batch])

        for sample_idx in batch:
            sentence, sentence_tag = self.samples[sample_idx].split("\t")
            num_paddings = max_sentence_length - len(sentence)
            sentence_input_vector = np.zeros((len(sentence) + num_paddings, self.num_chars), dtype=np.float32)
            sentence_output_vector = np.zeros((len(sentence) + num_paddings, self.num_pos), dtype=np.float32)

            for i, char in enumerate(sentence):
                if char in self.char_to_index:
                    char_index = self.char_to_index[char]
                else:
                    char_index = self.char_to_index["UNK"]
                sentence_input_vector[i, char_index] = 1

            for i, pos in enumerate(sentence_tag.split("/")[1:]):
                pos_index = self.pos_to_index[pos]
                sentence_output_vector[i, pos_index] = 1

            X.append(sentence_input_vector)
            y.append(sentence_output_vector)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        return X, y
