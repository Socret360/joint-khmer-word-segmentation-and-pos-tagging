# Joint Word Segmentation and POS Tagging in Keras

A Keras implementation of a deep learning network to simultaneously perform Word Segmentation and Part-of-Speech (POS) Tagging introduced by Bouy et al. in the paper [Joint Khmer Word Segmentation and Part-of-Speech Tagging Using Deep Learning](https://arxiv.org/abs/2103.16801).

## Requirements

```
tensorflow==2.7.0
```

## Config File Layout

```json
{
    "training": {
        "batch_size": 128, // The batch size during training
        "learning_rate": 0.001 // The learning rate
    },
    "model": {
        "num_stacks": 2, // The number of LSTM layer stacks.
        "hidden_layers_dim": 100 // The number of units for each hidden LSTM layers.
    }
}
```

## Training on Custom Dataset

### 1. Dataset Format

This repo expects datasets as text files in the below format. The sentence and sentence_tag are separated by a `\t` character.

```txt
sentence  sentence_tag
```

Sample:

```txt
ផលិត^កម្ម	/NN/NS/NS/NS/NS/NS/NS/NS/NS
នេះគឺ_ជាទេព្យផល្គុន	/DT/NS/NS/VB/NS/NS/NS/NS/PN/NS/NS/NS/NS/PN/NS/NS/NS/NS/NS
...
```

### 2. Start training

```cmd
python train.py config train_set char_map pos_map --shuffle=False --epochs=300 --output_dir=output
```

```txt
positional arguments:
  config                path to config file.
  train_set             path to training dataset.
  char_map              path to characters map file.
  pos_map               path to pos map file.

optional arguments:
  -h, --help                show this help message and exit.
  --shuffle [SHUFFLE]       whether to shuffle the dataset when creating the batch.
  --epochs EPOCHS           the number of epochs to train.
  --output_dir OUTPUT_DIR   path to output directory.
```

## Evaluating on Custom Dataset

### 1. Dataset Format

This repo expects datasets as text files in the below format. The sentence and sentence_tag are separated by a `\t` character.

```txt
sentence  sentence_tag
```

Sample:

```txt
ផលិត^កម្ម	/NN/NS/NS/NS/NS/NS/NS/NS/NS
នេះគឺ_ជាទេព្យផល្គុន	/DT/NS/NS/VB/NS/NS/NS/NS/PN/NS/NS/NS/NS/PN/NS/NS/NS/NS/NS
...
```

### 2. Start Evaluation Process

```cmd
python evaluate.py config test_set char_map pos_map weights --output_dir=output
```

```txt
positional arguments:
  config                path to config file.
  test_set              path to test dataset.
  char_map              path to characters map file.
  pos_map               path to pos map file.
  weights               path to weights file.

optional arguments:
  -h, --help                show this help message and exit
  --output_dir OUTPUT_DIR   path to output directory.
```

## About Pretrained Weights

You can access a pretrained weights [here](pretrained). The network was trained for 300 epochs on **khPOS's train.all2** dataset.

## Converting Pretrained Weights

You can convert the pretrained weights into a consolidated Keras format or tflite using the below command

```cmd
python convert.py config weights char_map pos_map --output_type=keras --output_dir=output
```
```txt
positional arguments:
  config                path to config file.
  weights               path to the weight file.
  char_map              path to characters map file.
  pos_map               path to pos map file.

optional arguments:
  -h, --help                  show this help message and exit.
  --output_dir OUTPUT_DIR     path to output directory.
  --output_type OUTPUT_TYPE   the type of the output model. One of type: "keras", "tflite"
```

## Pretrained Weights Evaluation

<table>
    <thead>
        <tr>
            <th>Test Set</th>
            <th>POS Tag</th>
            <th>Tag Accuracy (%)</th>
            <th>POS Tagging Accuracy (%)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=15>khPOS OPEN-TEST</td>
            <td>AB</td>
            <td>0.0</td>
            <td rowspan=15>91.75</td>
        </tr>
        <tr>
            <td>AUX</td>
            <td>95.91</td>
        </tr>
        <tr>
            <td>CC</td>
            <td>91.67</td>
        </tr>
        <tr>
            <td>CD</td>
            <td>94.06</td>
        </tr>
        <tr>
            <td>DT</td>
            <td>93.07</td>
        </tr>
        <tr>
          <td>IN</td>
          <td>94.12</td>
        </tr>
        <tr>
          <td>JJ</td>
          <td>67.4 </td>
        </tr>
        <tr>
          <td>VB</td>
          <td>88.67</td>
        </tr>
        <tr>
          <td>NN</td>
          <td>94.17</td>
        </tr>
        <tr>
          <td>PN</td>
          <td>91.06</td>
        </tr>
        <tr>
          <td>PA</td>
          <td>56.76</td>
        </tr>
        <tr>
          <td>PRO</td>
          <td>96.83</td>
        </tr>
        <tr>
          <td>QT</td>
          <td>0.0</td>
        </tr>
        <tr>
          <td>RB</td>
          <td>87.87</td>
        </tr>
        <tr>
          <td>SYM</td>
          <td>98.61</td>
        </tr>
        <tr>
            <td rowspan=15>khPOS CLOSE-TEST</td>
            <td>AB</td>
            <td>0.0</td>
            <td rowspan=15>95.78</td>
        </tr>
        <tr>
            <td>AUX</td>
            <td>99.49</td>
        </tr>
        <tr>
            <td>CC</td>
            <td>91.83</td>
        </tr>
        <tr>
            <td>CD</td>
            <td>98.91</td>
        </tr>
        <tr>
            <td>DT</td>
            <td>97.42</td>
        </tr>
        <tr>
          <td>IN</td>
          <td>97.15</td>
        </tr>
        <tr>
          <td>JJ</td>
          <td>81.13</td>
        </tr>
        <tr>
          <td>VB</td>
          <td>95.54</td>
        </tr>
        <tr>
          <td>NN</td>
          <td>98.53</td>
        </tr>
        <tr>
          <td>PN</td>
          <td>93.49</td>
        </tr>
        <tr>
          <td>PA</td>
          <td>64.79</td>
        </tr>
        <tr>
          <td>PRO</td>
          <td>98.97</td>
        </tr>
        <tr>
          <td>QT</td>
          <td>0.0</td>
        </tr>
        <tr>
          <td>RB</td>
          <td>89.42</td>
        </tr>
        <tr>
          <td>SYM</td>
          <td>99.35</td>
        </tr>
    </tbody>
</table>

## References

- Buoy, R., Taing, N., & Kor, S. (2021). Joint Khmer Word Segmentation and Part-of-Speech Tagging Using Deep Learning. Retrieved from <https://arxiv.org/abs/2103.16801>
- Loem, M. (2021, May 4). Joint Khmer Word Segmentation and POS tagging. Medium. Retrieved from <https://towardsdatascience.com/joint-khmer-word-segmentation-and-pos-tagging-cad650e78d30>
- Ye, K. T., Vichet, C., & Yoshinori, S. (2017). Comparison of Six POS Tagging Methods on 12K Sentences Khmer Language POS Tagged Corpus. First Regional Conference on Optical character recognition and Natural language processing technologies for ASEAN languages (ONA 2017). Retrieved from <https://github.com/ye-kyaw-thu/khPOS/blob/master/khpos.pdf>