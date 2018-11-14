"""Functions to work with demand networks."""

from tensorflow import keras
import tensorflow as tf
import numpy as np

tf.enable_eager_execution()


def build_model(number_node, number_good, optimizer, loss_fn, activation_fn='relu', metrics=None):
    """
    :param number_node: Integer or list of integers. Number of nodes in each hidden layer starting from the first hidden
    layer. If an integer is provided, the neural network has a single hidden layer. If a list of integers is provided,
    the number of hidden layers in the neural network is equal to the length of the list.
    :param number_good: Integer. Number of gooods. This entry is used to set the size of the output layer.
    :param optimizer: A keras.optimizers or a tf.train object to specify the optimization algorithm.
    :param loss_fn: A loss function to evaluate the model.
    :param activation_fn: Optional activation function for the hidden layers. Default: 'relu'. Alternatives:
    'None' for linear activation (or no activation), name of a built-in function from tensorflow.keras, or a function
    for hidden layer activation.
    :param metrics: Optional metric to evaluate during optimization. Default: None. This could be a function or a list
    of keras metric name.
    :return: model.
    model: Compiled neural network model with provided number of layers and nodes, hidden layer activation, and
    softmax output layer with #number_good# nodes. "model" also stores its optimizer and loss function for optimization.
    Example:
    optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)  # SGD from keras library.
    loss_fn = tf.keras.losses.MSE  # Mean squared error.
    metrics = ['mae']  #Mean absolute error.
    neural_network = build_model([10,5], 3, optimizer, loss_fn, activation_fn='sigmoid')
    """

    if isinstance(number_node, list) is False:
        number_node = [number_node]  # Turn input to list if not already a list.

    number_hidden_layer = number_node.__len__()

    model = keras.Sequential()  # Initialize model.
    hidden_layer = 0  # Initialize iterator for which hidden layer to add to the model.
    while hidden_layer < number_hidden_layer:  # Construct hidden layers.
        model.add(keras.layers.Dense(number_node[hidden_layer], activation=activation_fn))  # Add the hidden layer.
        hidden_layer += 1  # Move to the next layer.
    model.add(keras.layers.Dense(number_good, activation='softmax'))  # Construct the output softmax layer.

    model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)

    return model


def pd_to_tfdata(data, p_ident, e_ident, b_ident, d_ident=None, idx=None):
    """
    Converts a pandas dataframe to a tf.data.Dataset object. Only use this for small datasets. For large datasets,
    for example datasets that are >1GB, you may use the dataset in terms of placeholders. See
    https://www.tensorflow.org/guide/datasets -> Reading Input Data -> Consuming Numpy Arrays
    :param data: A pandas dataframe.
    :param p_ident: String identifier for price columns. Starting unique pattern of the column names for prices.
    :param e_ident: String identifier for expenditure columns. Starting unique pattern of the column name for
    expenditures.
    :param b_ident: String identifier for budget share columns. Starting unique pattern of the column names for budget
    shares.
    :param d_ident: Optional string identifier for other covariate columns. Default:None. If covariates other than
    prices and expenditures, for example demographics, are available, this is the starting unique pattern of the column
    names for the other covariates.
    :param idx: Optional list of indices. Default: None. If a sample from the provided data is selected for conversion,
    this is an array of indices to select observations using pandas.DataFrame row indices.
    :return: A tf.data.Dataset object with features and outputs.

    Example:
    tf.enable_eager_execution()  # To print output for this example
    d = {'p_good1': [1, 2], 'p_good2': [3, 4], 'exp': [5, 6], 'w_good1': [0.3, 0.7], 'w_good2':[0.7, 0.3]}
    data = pd.DataFrame(d)
    data
           p_good1  p_good2  exp  w_good1  w_good2
    0        1        3    5      0.3      0.7
    1        2        4    6      0.7      0.3

    # Without indices to select from
    tf_data = pd_to_tfdata(data, 'p', 'e', 'w')
    print(tf_data)
    <TensorSliceDataset shapes: ((3,), (2,)), types: (tf.int64, tf.float64)>
    tf_data.make_one_shot_iterator()
    for x, y in tf_data:
        print(x, y)
    tf.Tensor([1 3 5], shape=(3,), dtype=int64) tf.Tensor([0.3 0.7], shape=(2,), dtype=float64)
    tf.Tensor([2 4 6], shape=(3,), dtype=int64) tf.Tensor([0.7 0.3], shape=(2,), dtype=float64)

    # Selecting only the first row
    tf_data = pd_to_tfdata(data, 'p', 'e', 'w', idx=[0])
    print(tf_data)
    <TensorSliceDataset shapes: ((3,), (2,)), types: (tf.int64, tf.float64)>
    tf_data.make_one_shot_iterator()
    for x, y in tf_data:
        print(x, y)
    tf.Tensor([1 3 5], shape=(3,), dtype=int64) tf.Tensor([0.3 0.7], shape=(2,), dtype=float64)
    """

    # Extract data to work with using the provided indices.
    if idx is None:
        current_data = data
    else:
        current_data = data.loc[idx, :]

    # Extract input and output information: prices, expenditures, demographics as covariates and budget shares
    # as explanatory variables.
    price = current_data.loc[:, current_data.columns.str.startswith(p_ident)]  # Prices
    expenditure = current_data.loc[:, current_data.columns.str.startswith(e_ident)]  # Expenditures
    budget_share = current_data.loc[:, current_data.columns.str.startswith(b_ident)]  # Budget shares
    if d_ident is None:
        features = np.hstack([price.values, expenditure.values])  # Combine covariates to obtain the features array
    else:
        demographics = current_data.loc[:, current_data.columns.str.startswith(d_ident)]  # Demographics
        features = np.hstack([price.values, expenditure.values, demographics.values])
    outputs = budget_share.values

    # Create a tf.data.Dataset object
    dataset = tf.data.Dataset.from_tensor_slices((features, outputs))

    return dataset


# def obtain_features_outputs(idx, full_data, p_ident, e_ident, b_ident, d_ident=None):
#     """
#     Converts pandas DataFrames into demandtools::DemandDataset class that is needed for the optimizer's dataloader.
#     :param idx: Array of indices to pick from the full dataset.
#     :param full_data: Pandas DataFrame. Full dataset that includes all observations.
#     :param p_ident: String identifier for price columns. Starting unique pattern of the column names.
#     :param e_ident: String identifier for expenditure columns. Starting unique pattern of the column name.
#     :param b_ident: String identifier for budget share columns. Starting unique pattern of the column names.
#     :param d_ident: Optional. Default None. String identifier for demographics columns. Starting unique pattern of the
#     column names.
#     :return: data
#     data: Data of demandtools::DemandDataset class.
#     """
#     dataset_index = idx
#     dataset = full_data.loc[dataset_index, :]
#     price = dataset.loc[:, dataset.columns.str.startswith(p_ident)]
#     expenditure = dataset.loc[:, dataset.columns.str.startswith(e_ident)]
#     outputs = np.array(dataset.loc[:, dataset.columns.str.startswith(b_ident)])
#     if d_ident is None:
#         features = np.hstack([price.values, expenditure.values])  # Combine covariates to obtain the features array
#     else:
#         demographics = dataset.loc[:, dataset.columns.str.startswith(d_ident)]
#         features = np.hstack([price.values, expenditure.values, demographics.values])
#     return features, outputs
#
#
# class DataSequencer(tf.keras.utils.Sequence):
#     def __init__(self, features, outputs, batch_size):
#         self.x = np.array(features)
#         self.y = np.array(outputs)
#         self.batch_size = batch_size
#
#     def __len__(self):
#         """Number of batches per epoch"""
#         return int(self.x.shape[0] / self.batch_size)
#
#     def __getitem__(self, idx):
#         """Generate batch"""
#         batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
#         batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
#         return batch_x, batch_y
