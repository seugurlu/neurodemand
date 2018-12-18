"""Code to analyse cross-validation results."""

#import matplotlib.pyplot as plt # Uncomment to view figures on the fly.
import pandas as pd
import seaborn as sns
sns.set(style="darkgrid")

# Hyperparameters
file_path = "./output/analysis/cv_losses.csv"
row_index1 = "bootstrap_sample"
row_index2 = "n_node"
column_index = "cv_loss"
n_bootstrap = 200
figure_path = "./output/figures/"
n_nodes_csv_path = "./output/analysis/preferred_nodes.csv"

# Load cv losses
cv_losses = pd.read_csv(file_path).set_index([row_index1, row_index2])

# Store optimum number of nodes and cv losses for each sample.
bootstrap_range = range(n_bootstrap)
sample_keys = list(bootstrap_range)
optimum_n_node = pd.DataFrame(index=sample_keys, columns=[row_index2, column_index])
optimum_n_node.index.name = row_index1
for sample_key in bootstrap_range:
    optimum_n_node.loc[sample_key, row_index2] = cv_losses.loc[sample_key].idxmin().item()
    optimum_n_node.loc[sample_key, column_index] = cv_losses.loc[sample_key].min().item()
n_nodes_csv_path = optimum_n_node.to_csv(n_nodes_csv_path)

# Plot distribution of preferred number of nodes
preferred_nodes = optimum_n_node[row_index2]
ax = sns.countplot(preferred_nodes)
ax.set(xlabel="Number of Nodes", ylabel="Count")
fig = ax.get_figure()
figure_name = "cv_nodes_count_plot.pdf"
fig.savefig(figure_path+figure_name)
