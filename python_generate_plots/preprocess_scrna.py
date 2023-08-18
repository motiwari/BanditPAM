import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH="./data/single_cell/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices"
GC_MAT_PATH = DATA_PATH + '/filtered_matrices_mex/hg19'
ANALYSIS_10X_PATH = DATA_PATH + '/analysis_csv'
OUTPUT_PATH = DATA_PATH + '/NUMPY_OUT'

def load_10x(GC_MAT_PATH):
    '''
    https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/fresh_68k_pbmc_donor_a
    '''
    filename_data = GC_MAT_PATH + '/matrix.mtx'
    filename_genes = GC_MAT_PATH + '/genes.tsv'
    filename_barcodes = GC_MAT_PATH + '/barcodes.tsv'

    data = sc.read(filename_data, cache=True).transpose()
    data.var_names = np.genfromtxt(filename_genes, dtype=str)[:, 1]
    data.obs['barcode'] = np.genfromtxt(filename_barcodes, dtype=str)
    data.obs.index = data.obs['barcode']
    return data

# Load 10x 68k PBMC
adata = load_10x(GC_MAT_PATH)

# Load 10x analysis
df_pca = pd.read_csv(ANALYSIS_10X_PATH+'/pca/projection.csv', sep=',')
set_cell_keep = set(df_pca['Barcode'])

# Select cell consistent with the 10x analysis
ind_select = [x in set_cell_keep for x in adata.obs['barcode']]
adata = adata[ind_select, :]
# Select genes expressed in more than 100 cells
sc.pp.filter_genes(adata, min_cells=100)
# Size factor normalization (to have 1000 reads)
sc.pp.normalize_total(adata, target_sum=1000)
# log1p transform
sc.pp.log1p(adata)

# Save file as csv
temp_df = pd.DataFrame(index=adata.obs.index, columns=list(adata.var_names),
                       data=adata.X.todense())
temp_df.to_csv(OUTPUT_PATH+'/data.csv.gz', sep=',', compression='gzip')

# See if the file can be read
temp_df_ref = pd.read_csv(OUTPUT_PATH+'/data.csv.gz',
                          sep=',', compression='gzip', index_col=0)

# Should be 0
print((np.absolute(temp_df_ref.values-temp_df.values)>1e-5).sum())
