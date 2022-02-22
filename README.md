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
# References
- Buoy, R., Taing, N., & Kor, S. (2021). Joint Khmer Word Segmentation and Part-of-Speech Tagging Using Deep Learning. arXiv preprint arXiv:2103.16801.
- Loem, M. (2021, May 4). Joint Khmer Word Segmentation and POS tagging. Medium. Retrieved February 22, 2022, from https://towardsdatascience.com/joint-khmer-word-segmentation-and-pos-tagging-cad650e78d30