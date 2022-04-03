from tensorflow.keras.models import Model,  Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Activation, Input, Flatten


def Network(output_dim, embedding_dim, num_stacks, hidden_layers_dim, batch_size=128, max_sentence_length=None) -> Model:
    """ Defines the structure of the network.

    Args
    ---
    - output_dim: An int representing the size of the output vector.
    - embedding_dim: An int representing the size of the character vector.
    - num_stacks: An int representing the number of LSTM stacks.
    - batch_size: An int representing the batch size. (Defaults to 128)
    - max_sentence_length: An int representing the max length of a sentence. (Defaults to None)

    Returns
    ---
    Keras model object.

    Raises
    ---
    - `output_dim` must be larger than 0
    - `embedding_dim` must be larger than 0
    - `num_stacks` must be larger than 0
    - `batch_size` must be larger than 0
    - `hidden_layers_dim` must be larger than 0

    References
    ---
    - Buoy, R., Taing, N., & Kor, S. (2021). Joint Khmer Word Segmentation and Part-of-Speech Tagging Using Deep Learning. arXiv preprint arXiv:2103.16801.
    - Loem, M. (2021, May 4). Joint Khmer Word Segmentation and POS tagging. Medium. Retrieved February 22, 2022, from https://towardsdatascience.com/joint-khmer-word-segmentation-and-pos-tagging-cad650e78d30
    """
    assert output_dim > 0, "output_dim must be larger than 0"
    assert embedding_dim > 0, "embedding_dim must be larger than 0"
    assert num_stacks > 0, "num_stacks must be larger than 0"
    if batch_size is not None:
        assert batch_size > 0, "batch_size must be larger than 0"
    assert hidden_layers_dim > 0, "hidden_layers_dim must be larger than 0"

    layers = [
        Input(shape=(max_sentence_length, embedding_dim), batch_size=batch_size),
        Bidirectional(LSTM(hidden_layers_dim, return_sequences=True)),
    ]
    for i in range(num_stacks - 1):
        layers.append(Bidirectional(LSTM(hidden_layers_dim, return_sequences=i != num_stacks-1)))

    layers.append(Dense(output_dim))
    layers.append(Activation("softmax"))

    return Sequential(layers)

    # input_layer = Input(shape=(max_sentence_length, embedding_dim), batch_size=batch_size)
    # x = Bidirectional(LSTM(hidden_layers_dim, return_sequences=True))(input_layer)
    # for i in range(num_stacks - 1):
    #     x = Bidirectional(LSTM(hidden_layers_dim, return_sequences=i != num_stacks-1))(x)
    # x = Dense(output_dim)(x)
    # x = Activation("softmax")(x)
    # model = Model(inputs=input_layer, outputs=x)
    # return model
