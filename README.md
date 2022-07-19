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

The contacts for each cell is an individual file where each line represents a chromatin contact.
The formart is `chrom1\tposition1\tchrom2\tposition2`. The `position` could be the midpoint of a fragment.



Three types of raw data are required to generate genomic mutation matrix, gene expression matrix and DNA methylation matrix from CCLE database.


`CCLE_mutations.csv` - Genomic mutation profile from CCLE database

`CCLE_expression.csv` - Gene expression profile from CCLE database

`CCLE_RRBS_TSS_1kb_20180614.txt` - DNA methylation profile from CCLE database

The three types of raw data `genomic mutation file`, `gene expression file` and `DNA methylation file` can be downloaded from [CCLE database](https://depmap.org/portal/download/) or from our provided [Cloud Server](https://cloud.tsinghua.edu.cn/d/9b42762d8eef4f42a835/). 

After data preprocessed, the three following preprocessed files will be in located in `data` folder.

`genomic_mutation_34673_demap_features.csv` --  genomic mutation matrix where each column denotes mutation locus and each row denotes a cell line

`genomic_expression_561celllines_697genes_demap_features.csv` -- gene expression matrix where each column denotes a coding gene and each row denotes a cell line

`genomic_methylation_561celllines_808genes_demap_features.csv` -- DNA methylation matrix where each column denotes a methylation locus and each row denotes a cell line

We recommend to start from the preprocessed data. Please note that each preprocessed file is in `csv` format, of which the column and row name are provided to speficy `mutation location`, `gene name`, `methylation location` and corresponding `Cell line`.

**Step 2: Drug feature representation**

Each drug in our study will be represented as a graph containing nodes and edges. From the GDSC database, we collected 223 drugs that have unique Pubchem ids. Note that a drug under different screening condition (different GDSC drug id) may share the same Pubchem id.
Here, we used [deepchem](https://github.com/deepchem/deepchem) library for extracting node features and gragh of a drug. The node feature  (75 dimension) corresponds to a stom in within a drug, which includes atom type, degree and hybridization, etc. 

We recorded three types of features in a list as following

```python
drug_feat = [node_feature, adj_list, degree_list]
node_feature - features of all atoms within a drug with size (nb_atom, 75)
adj_list - adjacent list of all atoms within a drug. It denotes the all the neighboring atoms indexs
degree_list - degree list of all atoms within a drug. It denotes the number of neighboring atoms 
```

The above feature list will be further compressed as `pubchem_id.hkl` using hickle library.

Please note that we provided the extracted features of 223 drugs from GDSC database, just unzip the `drug_graph_feat.zip` file in `data/GDSC` folder


**Step 3: DeepCDR model training and testing**

Here, we provide both DeepCDR regression and classification model here.

### DeepCDR regression model

```python
python run_DeepCDR.py -gpu_id [gpu_id] -use_mut [use_mut] -use_gexp [use_gexp] -use_methy [use_methy] 
[gpu_id] - set GPU card id (default:0)
[use_mut] - whether use genomic mutation data (default: True)
[use_gexp] - whether use gene expression data (default: True)
[use_methy] - whether use DNA methylation data (default: True)
```
One can run `python run_DeepCDR.py -gpu_id 0 -use_mut True -use_gexp True -use_methy True` to implement the DeepCDR regression model.

The trained model will be saved in `data/checkpoint` folder. The overall Pearson's correlation will be calculated.

### DeepCDR classification model

```python
python run_DeepCDR_classify.py -gpu_id [gpu_id] -use_mut [use_mut] -use_gexp [use_gexp] -use_methy [use_methy] 
[gpu_id] - set GPU card id (default:0)
[use_mut] - whether use genomic mutation data (default: True)
[use_gexp] - whether use gene expression data (default: True)
[use_methy] - whether use DNA methylation data (default: True)
```
One can run `python run_DeepCDR_classify.py -gpu_id 0 -use_mut True -use_gexp True -use_methy True` to implement the DeepCDR classification model.

The trained model will be saved in `data/checkpoint` folder. The overall auROC and auPRC will be calculated.

## External patient data

We also provided the external patient data downloaded from [Firehose Broad GDAC](http://gdac.broadinstitute.org/runs/stddata__2016_01_28/). The patient data were preprocessed the same way as cell line data. The preprocessed data can be downloaded from our [Server](https://cloud.tsinghua.edu.cn/f/f0d3420e712c43c9a688/). 

The preprocessed data contain three important files:

`mut.csv` - Genomic mutation profile of patients

`expr.csv` - Gene expression profile of patients

`methy.csv` - DNA methylation profile of patients

Note that the preprocessed patient data (`csv` format) have exact the same columns names as the three cell line data (`genomic_mutation_34673_demap_features.csv`, `genomic_expression_561celllines_697genes_demap_features.csv`, `genomic_methylation_561celllines_808genes_demap_features.csv`). The only difference is that the row name of patient data were replaced with patient unique barcode instead of cell line name.

Such format-consistent data is easy for external evaluation by repacing the cell line data with patient data.


# Contact
If you have any question regard our code or data, please do not hesitate to open a issue or directly contact me (liuqiao@stanford.edu)

# License
This project is licensed under the MIT License - see the LICENSE.md file for details


























