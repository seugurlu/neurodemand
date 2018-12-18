from shutil import copy
import pandas as pd

#Hyperparameters
n_bootstrap = 200

# Import preferred numbers of nodes
file_path = "./output/analysis/preferred_nodes.csv"
index_col = 'bootstrap_sample'
preferred_n_nodes = pd.read_csv(file_path, index_col=index_col)
col_name = "n_node"
source_path = "./output/temp/dn/cross_validation/"
destination_path = "./output/estimation/dn/"

for sample_key in range(n_bootstrap):
	n_node = preferred_n_nodes.loc[sample_key, col_name]
	source_name = "sample_"+str(sample_key)+"_node_"+str(n_node)+".h5"
	source = source_path + source_name
	destination_name = "sample_"+str(sample_key)+".h5"
	destination = destination_path + destination_name
	copy(source, destination)
