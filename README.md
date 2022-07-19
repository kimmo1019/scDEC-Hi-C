# scDEC-Hi-C
Deep generative modeling and clustering of single cell Hi-C data.
 
![model](https://github.com/kimmo1019/scDEC-Hi-C/blob/main/model.png)
 
scDEC-Hi-C is a novel end-to-end deep learning framework for analyzing single cell Hi-C data using a multi-stage model. scDEC-Hi-C consists of a chromosome-wise autoencoder (AE) model and a cell-wise deep embedding and clustering model (scDEC).

 
 # Requirements
- Keras==2.1.4
- TensorFlow==1.13.1

# Installation
scDEC-Hi-C can be downloaded by
```shell
git clone https://github.com/kimmo1019/scDEC-Hi-C
```
Installation has been tested in a Linux/MacOS platform.

# Instructions
We provide detailed step-by-step instructions for running scDEC-Hi-C model including data preprocessing, model training, and model test.

## Model implementation

**Step 1: Data Preparing**

The contacts for each cell is an individual file where each line represents a chromatin contact saved in `datasets` folder.

The formart is `chrom1\tposition1\tchrom2\tposition2`. The `position` could be the midpoint of a fragment.

**Step 2: Model input preparing**

In the `preprocess` folder, we provide script to get the model input from the raw contact files.

Users could run the following script for getting model input.

```python
python get_mats.py --name [name] --genome [genome] -res [res] -size [size] 
[name] - dataset name (default: Ramani)
[genome] - which genome to use (default: hg19)
[res] - resolution for single cell Hi-C data (default: 1000000)
[size] - size for constructing matrix (default: 28)
```

After running, `data_norm.pkl` and `data_resize_[size].npy` will be saved in the corresponding data folder.

**Step 3: scDEC-Hi-C model training and testing**


### scDEC-Hi-C training

The code for running scDEC-Hi-C on Ramani dataset should be the following:

```python
python main_clustering.py --data Ramani --K 4 --dx 10 --dy 1150 --alpha 10 --beta 10 --bs 32 --train True
[data] - dataset name (e.g., Ramani)
[K] - number of clusters/cell types 
[dx] - dimension of latent variable z in the paper
[dy] - dimension of concatenated chromosome
[alpha] - coefficient for roundtrip loss
[beta] - coefficient for cross-entropy loss
```

After running, the results will be stored in `results` folder. `data_at_[batch_id].npz` will be saved, which stores `data_x_` and `data_x_onehot_`.

`data_x_`: reconstructed latent variable z.
`data_x_onehot_`: reconstructed Categorical variable c.


# Contact
If you have any question regard our code or data, please do not hesitate to open a issue or directly contact me (liuqiao@stanford.edu)

# License
This project is licensed under the MIT License - see the LICENSE.md file for details


























