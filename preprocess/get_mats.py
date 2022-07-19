import numpy as np
import os,sys
import pickle
import argparse


def get_hic_mat_per_cell(contacts_dir, each_file, res):
    matrix = {}
    for key in chrom2len:
        mat_size = chrom2len[key] // res + 1
        matrix[key] = np.zeros((mat_size,mat_size))
    for line in open('%s/%s'%(contacts_dir,each_file)).readlines():
        chrom1, pos1, chrom2, pos2 = line.strip().split('\t')
        if chrom1 != chrom2:
            continue
        mat_size = chrom2len[chrom1] // res + 1
        idx1 = int(pos1) // res
        idx2 = int(pos2) // res
        matrix[chrom1][idx1, idx2] += 1
        if idx1 != idx2:
            matrix[chrom1][idx2, idx1] += 1
    return matrix

def get_hic_mat(contacts_dir, res=1000000):
    data = []
    cell_labels=[]
    cell_barcodes = []
    files_list = []
    for each_file in os.listdir(contacts_dir):
        if each_file[-8:] == 'contacts':
            files_list.append(each_file.split('.')[0])
            cell_label = each_file.split('_')[0]
            cell_barcode = each_file.split('_')[1].split('.')[0]
            cell_labels.append(cell_label)
            cell_barcodes.append(cell_barcode)
            matrix = get_hic_mat_per_cell(contacts_dir, each_file ,res)
            data.append(matrix)
    print('Stats:',[(item, list(cell_labels).count(item)) for item in np.unique(cell_labels)])
    return data, cell_labels, cell_barcodes, files_list

def normalize(data, cell_labels, cell_barcodes, files_list):
    if name == 'Ramani':
        all_max_nb_contacts = [int(open('../datasets/%s/higashi_contact_pairs/%s_%s.meta'%(name,item[0],item[1])).readline().split('\t')[0]) \
            for item in list(zip(cell_labels,cell_barcodes))]
    else:
        all_max_nb_contacts = [int(open('../datasets/%s/contact_pairs/%s.meta'%(name, item)).readline().split('\t')[0]) \
            for item in files_list]
    for i in range(len(data)):
        for key in data[i].keys():
            data[i][key] = np.log(data[i][key]*(np.max(all_max_nb_contacts)/all_max_nb_contacts[i])+1)
    return data

def resize(data,size=50):
    import cv2
    chroms = list(data[0].keys())
    data_resize = np.empty((len(data),size,size,len(chroms)))
    for i in range(len(data)):
        for j in range(len(chroms)):
            data_resize[i,:,:,j] = cv2.resize(np.array(data[i][chroms[j]],dtype='float32'),(size, size))
    return data_resize
    
def save_file(data, path):
    f = open(path, 'wb')
    pickle.dump(data, f, protocol = 2)
    f.close()

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
    raw_data_path = '%s/data_res_%d.pkl'%(save_dir, res)
    norm_data_path = '%s/data_norm_res_%d.pkl'%(save_dir, res)

    if os.path.exists(raw_data_path):
        data, cell_labels, cell_barcodes = pickle.load(open(raw_data_path,'rb'))
    else:
        data, cell_labels, cell_barcodes, files_list = get_hic_mat(contacts_dir=contacts_dir, res=res)
        nb_cells = len(data)
        save_file([data, cell_labels, cell_barcodes, files_list], raw_data_path)
    
    if os.path.exists(norm_data_path):
        data_norm, cell_labels, cell_barcodes, files_list = pickle.load(open(norm_data_path,'rb'))
    else:
        data_norm = normalize(data, cell_labels, cell_barcodes, files_list)
        save_file([data_norm, cell_labels, cell_barcodes, files_list], norm_data_path)
    
    for i in range(4):
        data_resize = resize(data_norm,size*2**i)
        np.save('%s/data_resize_%d.npy'%(save_dir, size*2**i), data_resize)
        

