# Joint Word Segmentation and POS Tagging in Keras

## Requirements

```txt
tensorflow==2.7.0
```

## Training

```cmd
python train.py config train_set char_map pos_map --shuffle=False --epochs=100 --output_dir=output
```

```txt
positional arguments:
  config                path to config file.
  train_set             path to training dataset.
  char_map              path to characters map file.
  pos_map               path to pos map file.

optional arguments:
  -h, --help                show this help message and exit
  --shuffle [SHUFFLE]       whether to shuffle the dataset when creating the batch
  --epochs EPOCHS           the number of epochs to train
  --output_dir OUTPUT_DIR   path to output directory.
```

## Using Pretrained Model

Model train on khPOS train.all2

## Pretrained Model Evaluation

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

- Buoy, R., Taing, N., & Kor, S. (2021). Joint Khmer Word Segmentation and Part-of-Speech Tagging Using Deep Learning. arXiv preprint arXiv:2103.16801.
- Loem, M. (2021, May 4). Joint Khmer Word Segmentation and POS tagging. Medium. Retrieved February 22, 2022, from <https://towardsdatascience.com/joint-khmer-word-segmentation-and-pos-tagging-cad650e78d30>
