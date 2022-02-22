# Usage
## Requirements
```txt
tensorflow==2.7.0
```

## Training
```cmd 
python train.py config train_set char_map pos_map --shuffle=False --epochs=100 --output_dir=output
```

```txt
Start the training process.

positional arguments:
  config                path to config file.
  train_set             path to training dataset.
  char_map              path to characters map file.
  pos_map               path to pos map file.

optional arguments:
  -h, --help            show this help message and exit
  --shuffle [SHUFFLE]   whether to shuffle the dataset when creating the batch
  --epochs EPOCHS       the number of epochs to train
  --output_dir OUTPUT_DIR
                        path to output directory.
```
## Using Pretrained Model
Model train on khPOS train.all2

## Pretrained Model Evaluation
Test Set: khPOS OPEN-TEST | **POS Tag Accuracy: 91.75%**
| POS TAG | Tag Acc |
| :---:   | :---:   |
| AB      | 0.0     |
| AUX     | 95.91   | 
| CC      | 91.67   |
| CD      | 94.06   |
| DT      | 93.07   |
| IN      | 94.12   |
| JJ      | 67.4    |
| VB      | 88.67   |
| NN      | 94.17   |
| PN      | 91.06   |
| PA      | 56.76   |
| PRO     | 96.83   |
| QT      | 0.0     |
| RB      | 87.87   |
| SYM     | 98.61   |

Test Set: khPOS CLOSE-TEST | **POS Tag Accuracy: 95.78%**
| POS TAG | Tag Acc |
| :---:   | :---:   |
| AB      | 0.0     |
| AUX     | 99.49   |
| CC      | 91.83   |
| CD      | 98.91   |
| DT      | 97.42   |
| IN      | 97.15   |
| JJ      | 81.13   |
| VB      | 95.54   |
| NN      | 98.53   |
| PN      | 93.49   |
| PA      | 64.79   |
| PRO     | 98.97   |
| QT      | 0.0     |
| RB      | 89.42   |
| SYM     | 99.35   |

# References
- Buoy, R., Taing, N., & Kor, S. (2021). Joint Khmer Word Segmentation and Part-of-Speech Tagging Using Deep Learning. arXiv preprint arXiv:2103.16801.
- Loem, M. (2021, May 4). Joint Khmer Word Segmentation and POS tagging. Medium. Retrieved February 22, 2022, from https://towardsdatascience.com/joint-khmer-word-segmentation-and-pos-tagging-cad650e78d30