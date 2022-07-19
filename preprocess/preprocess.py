import scHiCTools
from scHiCTools import scHiCs
import numpy as np
import os,sys
import pickle
import argparse



def load_data(files, ref_genome, resolution, binary):
    loaded_data = scHiCs(files, 
                        reference_genome=ref_genome,
                        resolution=resolution, 
                        keep_n_strata=10,
                        format='shortest', adjust_resolution=True,
                        store_full_map=True,
                        customized_format=1234, header=0, chromosomes='except Y',
                        operations=['log2','convolution','random_walk'],
                        sep='\t'
                        )
    if binary:
        matrix = []
        for ch in y.chromosomes:
            A = y.full_maps[ch].copy()
            A.shape = (A.shape[0],A.shape[1]*A.shape[2])
            thres = np.percentile(A, 80, axis=1)
            A = (A > thres[:,None])
            pca = PCA(n_components = 20)
            R_reduce = pca.fit_transform(A)
            matrix.append(R_reduce)
        matrix = np.concatenate(matrix, axis=1)
        print('matrix.shape:', matrix.shape)

        pca = PCA(n_components = 100)
        matrix_reduce = pca.fit_transform(matrix)
        ding_schicluster_100d.npy",matrix_reduce[:,:100])



    

if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('--name', type=str, default='Ramani',help='name of dataset')
    parser.add_argument('--genome', type=str, default='hg19',help='genome building')
    parser.add_argument('--path', type=str, default='..',help='project path')
    parser.add_argument('--res', type=int, default=1000000,help='resolution')
    parser.add_argument('--size', type=int, default=28,help='project path')
    args = parser.parse_args()
    name = args.name
    genome = args.genome
    path = args.path
    res = args.res
    size = args.size

    chrom2len = {item.split('\t')[0]:int(item.strip().split('\t')[-1]) for item in open("%s.chrom.sizes"%genome).readlines()}
    contacts_dir = '%s/datasets/%s/contact_pairs' %(path, name)
    save_dir = '%s/datasets/%s'%(path, name)

    files = [item.rstrip() for item in open('/home/users/liuqiao/work/Higashi/data/%s/filelist.txt'%name).readlines()]
    print('%d cells were chosen for preprocessing'%len(files))
    load_data(files, ref_genome=genome, resolution=res, binary=True)
