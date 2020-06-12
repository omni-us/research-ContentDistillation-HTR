[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)

# Distilling Content from Style for Handwritten Word Recognition

A novel method that is able to disentangle the content and style aspects of input images by jointly optimizing a generative process and a handwritten word recognizer.

![Architecture](https://user-images.githubusercontent.com/9562709/78990677-5b18ef80-7b37-11ea-8347-cab821f154cc.png)


[Distilling Content from Style for Handwritten Word Recognition]()<br>
Lei Kang, Pau Riba, Marçal Rusiñol, Alicia Fornés, and Mauricio Villegas<br>
Accepted to ICFHR2020


## Software environment

- Ubuntu 16.04 x64
- Python 3
- PyTorch 1.0.1

## Dataset preparation

We carry on our experiments on the widely used handwritten dataset [IAM](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database).

## Note before running the code

- The training takes a lot of GPU memory, in my case, it takes 24GB in a GPU RTX 6000 with batchsize 8. Even if we set the basesize to 1, it still takes 16GB GPU memory. 

## How to train?

Once the dataset is prepared, you need to denote the correct urls in the file `load_data.py`, then you are ready to go. To run from scratch:
```
./run_train_scratch.sh
```

Or to start with a saved checkpoint:
```
./run_train_pretrain.sh
```
**Note**: Which GPU to use or which epoch you want to start from could be set in this shell script. (Epoch ID corresponds to the weights that you want to load in the folder `save_weights`)

## How to test?

```
./run_test.sh
```
And don't forget to change the epoch ID in this shell script to load the correct weights of the model that is corresponding to the epoch ID.

## Citation

If you use the code for your research or application, please cite our paper:

```
To be filled.
```
