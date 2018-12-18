import numpy as np
import demandtools as dt
import torch as th
import torch.nn as nn
import torch.utils.data as utils_data


"""Helper functions for demand estimators."""


def turn_to_dataset_class(idx, full_data, p_ident, e_ident, b_ident, d_ident=None):
    """
    Converts pandas DataFrames into demandtools::DemandDataset class that is needed for the optimizer's dataloader.
    :param idx: Array of indices to pick from the full dataset.
    :param full_data: Pandas DataFrame. Full dataset that includes all observations.
    :param p_ident: String identifier for price columns. Starting unique pattern of the column names.
    :param e_ident: String identifier for expenditure columns. Starting unique pattern of the column name.
    :param b_ident: String identifier for budget share columns. Starting unique pattern of the column names.
    :param d_ident: Optional. Default None. String identifier for demographics columns. Starting unique pattern of the
    column names.
    :return: data
    data: Data of demandtools::DemandDataset class.
    """
    dataset_index = idx
    dataset = full_data.loc[dataset_index, :]
    price = dataset.loc[:, dataset.columns.str.startswith(p_ident)]
    expenditure = dataset.loc[:, dataset.columns.str.startswith(e_ident)]
    budget_share = dataset.loc[:, dataset.columns.str.startswith(b_ident)]
    if d_ident is not None:
        demographics = dataset.loc[:, dataset.columns.str.startswith(d_ident)]
        data = dt.DemandDataset(price, expenditure, budget_share, demographics)
    else:
        data = dt.DemandDataset(price, expenditure, budget_share)
    return data


def pull_bs_sample(sample_key, idx, full_data, p_ident, e_ident, b_ident, d_ident=None):
    """
    Retrieves a particular bootstrap sample from the training sample using idx.
    :param sample_key: Integer bootstrap sample id.
    :param idx: Dictionary of indices for training, cross-validation and test samples.
    :param full_data: Pandas Dataframe. Data frame to draw bootstrap training samples.
    :param p_ident: String identifier for price columns. Starting unique pattern of the column names.
    :param e_ident: String identifier for expenditure columns. Starting unique pattern of the column name.
    :param b_ident: String identifier for budget share columns. Starting unique pattern of the column names.
    :param d_ident: String identifier for demographics columns. Starting unique pattern of the column names.
    :return: training_sample, cv_sample
    training_sample: Bootstrap training dataset of class demandtools::DemandDataset. Needed to pass into
    th.utils.dataloader.
    cv_sample: Bootstrap cross-validation dataset of class demandtools::DemandDataset. In order to pass into
    th.utils.dataloader.
    """
    training_sample = turn_to_dataset_class(idx[sample_key]['training_sample'], full_data,
                                            p_ident, e_ident, b_ident, d_ident)
    cv_sample = turn_to_dataset_class(idx[sample_key]['cv_sample'], full_data,
                                      p_ident, e_ident, b_ident, d_ident)
    return training_sample, cv_sample


def set_model(n_input_layer, n_hidden_layer, n_output_layer, activation_function, device='cpu'):
    """
    Calls demandtools::network_model to define a demand model with given parameters.
    :param n_input_layer: Integer. Number of nodes at the input layer.
    :param n_hidden_layer: Integer or tuple of integers. Number of nodes at the hidden layers.
    :param n_output_layer: Integer. Number of nodes at the output layer (number of goods).
    :param activation_function: Activation function to use in hidden layers for non-linear transformation.
    :param device: Optional. Default is 'cpu'. Whether the model should be kept at cpu or transferred to gpu.
    Pass cuda device here if gpu is used for estimation. Model is then transferred to that device.
    :return: nn_model
    nn_model: Network model with given parameters.
    """
    nn_model = dt.network_model(n_input_layer, n_hidden_layer, n_output_layer, activation_function)
    nn_model = nn.Sequential(nn_model)
    if device != 'cpu':
        nn_model.to(device)
    return nn_model


def dn_elasticities(x, n_good, nn_model):
    """
    Calculates income elasticities, uncompensated price elasticities, price elasticities, and slutsky matrices for all
    observations in x.
    :param x: Tensor. n_obs-by-n_good+n_demographic+1 tensor of input data.
    :param n_good: Integer. Number of goods.
    :param nn_model: Pytorch neural network model to predict outputs given inputs x.
    :return: income_elasticity, uncompensated_price_elasticity, compensated_price_elasticity, slutsky_matrix
    income_elasticity: Tensor. n_obs-by-n_good tensor of income elasticities.
    uncompensated_price_elasticity: Tensor. n_obs-by-n_good-by-n_good tensor of uncompensated price elasticities.
    compensated_price_elasticity: Tensor. n_obs-by-n_good-by-n_good tensor of compensated price elasticities.
    slutsky_matrix: Tensor. n_obs-by-n_good-by-n_good tensor of slutsky matrix.
    """
    # Initialize
    n_obs = x.shape[0]

    #Compute mu_ij and mu_i
    x.requires_grad_(True)
    y_pred = nn_model(x)
    mu_i = th.zeros([n_obs, n_good])
    mu_ij = th.zeros([n_obs, n_good, n_good])
    gradients = th.ones(n_obs)
    for good in range(n_good):
        y_pred[:, good].backward(gradients, retain_graph=True)
        derivatives = x.grad.clone()
        mu_i[:, good] = derivatives[:, n_good]
        mu_ij[:, good, :] = derivatives[:, :n_good]
        x.grad.zero_()

    #Calculate income elasticities
    income_elasticity = mu_i/y_pred + 1

    #Calculate uncompensated price elasticities
    kronecker_delta = th.eye(n_good)
    uncompensated_price_elasticity = mu_ij / y_pred[:,:,None].expand(-1,-1,10) - \
                                     kronecker_delta.repeat(n_obs, 1).reshape(n_obs, n_good, n_good)

    #Calculate compensated price elaticity
    compensated_price_elasticity = uncompensated_price_elasticity + income_elasticity[:,:,None]*y_pred[:,None,:]

    #Calculate Slutsky Matrix
    slutsky_matrix = y_pred[:,:,None]*compensated_price_elasticity

    return income_elasticity, uncompensated_price_elasticity, compensated_price_elasticity, slutsky_matrix


def theoretical_cost(yhat, slutsky_matrix, symmetry=True, negativity=True):
    # Initialize
    if yhat.dim() < 2:
        yhat = yhat[None, :]
    elif yhat.dim() > 2:
        raise ValueError("yhat must be n_obs-by-n_good but has more than 2 dimensions.")

    n_obs, n_good = yhat.shape

    input_check = slutsky_matrix.dim()
    if input_check < 2:
        raise ValueError("slutsky_matrix must be 2 or 3 dimensional.")
    elif input_check == 2:
        slutsky_matrix = slutsky_matrix[None, :, :]  # This is because the following code assumes dimension 1 is obs.

    if symmetry is True:
        # Calculate mean squared deviation from symmetry
        slutsky_matrix_transpose = th.einsum('ikj', [slutsky_matrix])
        symmetry_sq_dif = (slutsky_matrix - slutsky_matrix_transpose)**2
        symmetry_cost = 0.5*th.einsum('ijk->i', [symmetry_sq_dif])
        mean_symmetry_cost = symmetry_cost.mean()
    else:
        mean_symmetry_cost = 0

    if negativity is True:  # TODO: Test this whole block with a single observation. Something feels wrong.
        # Expenditure matrix has to be concave, so -slutsky matrix to feed into quasiconvexity check.
        A = -slutsky_matrix
        u = yhat - th.cat((th.norm(yhat, dim=1, keepdim=True), th.zeros([n_obs, n_good - 1])), dim=1)
        beta = -0.5*(u**2).sum(dim=1, keepdim=True)
        Au = th.einsum('ijk,ikl->ijl', (A, u[:, :, None]))
        u_tAu = th.einsum('ijk,ikl->ijl', (u[:, None, :], Au)).squeeze(dim=2)
        alpha = beta**(-2)*u_tAu
        w = -beta[:, :, None]**(-1)*Au
        alpha_div2_u = 0.5*alpha*u
        v = alpha_div2_u[:, :, None] - w
        v_t = th.einsum('ikj', [v])
        K = A + th.einsum('ijk,ikl->ijl', (u[:, :, None], v_t)) + th.einsum('ijk,ikl->ijl', (v, u[:, None, :]))
        K22 = K[:, 1:, 1:]
        mean_negativity_cost = 0
        for i in range(n_obs):
            # This part eliminates complex eigenvalues. So selection is done over real parts. A lexicographic ordering
            # would be more suitable but it does not exist in Pytorch as of Pytorch 0.41.
            eigenvalues = K22[i].eig()[0][:, 0]  # TODO: Create Lexicographic ordering if a future Pytorch supports it.
            selection = th.Tensor([0, eigenvalues.min()])
            # If the min eigenvalue is negative, then expenditure function's concavity is violated.
            mean_negativity_cost = mean_negativity_cost + th.min(selection)**2 / n_obs
    else:
        mean_negativity_cost = 0

    return mean_symmetry_cost, mean_negativity_cost


def dn_estimation(sample_key, full_data, idx_bootstrap, p_ident, e_ident, b_ident,
                  nn_model, optimizer, loss_fn,
                  d_ident=None, max_tol=1e-8,
                  max_iter=1000, batch_size=256, batch_size_factor=2, device='cpu', print_output=True, follow_cv=True):
    """
    Demand network optimizer.
    :param sample_key: Integer. Which bootstrap sample to use as the training sample.
    :param full_data: Pandas DataFrame. Main data frame to draw bootstrap samples.
    :param idx_bootstrap: Dictionary of indices for training, cross-validation and test samples.
    :param p_ident: String identifier for price columns. Starting unique pattern of the column names.
    :param e_ident: String identifier for expenditure columns. Starting unique pattern of the column name.
    :param b_ident: String identifier for budget share columns. Starting unique pattern of the column names.
    :param nn_model: Neural network model object. Obtained from set_model.
    :param optimizer: Neural network optimizer. Set with torch.optim.
    :param loss_fn: Loss function. Set with th.nn.
    :param d_ident: Optional String. Default: None. String identifier for demographics columns. Starting unique pattern
    of the column names.
    :param max_tol: Optional Float. Default: 1e-8. Maximum tolerance for convergence.
    :param max_iter: Optional Integer. Default: 200. Maximum number of iterations before increase in batch-size.
    :param batch_size: Optional Integer. Default: 32. Starting batch size. Capped at number of observations.
    :param batch_size_factor: Optional Integer. Default: 2. Factor of increase in batch_size if convergence is not
    achieved. New batch_size is batch_size*batch_size_factor if sample size is not exhausted. Batch size is capped at
    number of observations.
    :param device: Optional String. Default: 'cpu'. Which device to use for optimization.
    :param print_output: Optional Bool. Default: True. Should optimization print information during epochs?
    :param follow_cv: Optional Bool. Default: True. Should optimization follow evolution of cross-validation cost during
    epochs? If set to False, cross-validation cost after convergence is reported.
    :return: nn_model, loss_matrix, cv_loss.
    nn_model: Trained version of the input nn_model.
    loss_matrix: Array of training loss after each iteration.
    cv_loss: If follow_cv is True, an array of cross-validation loss after each iteration. Else, a float that reports
    cross-validation cost of trained model.
    """
    # Obtain bootstrap sample
    training_data, cv_data = pull_bs_sample(sample_key, idx_bootstrap, full_data, p_ident, e_ident, b_ident, d_ident)
    x_train = training_data.x
    y_train = training_data.y
    x_cv = cv_data.x
    y_cv = cv_data.y

    if device != 'cpu':
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        x_cv = x_cv.to(device)
        y_cv = y_cv.to(device)

    # Set Optimization Hyper-parameters
    n_obs, n_good = y_train.shape
    iteration = 0
    loss_matrix = th.zeros([max_iter], device=device)
    cv_loss_matrix = th.zeros([max_iter], device=device)
    tol = 1e5
    convergence = 0
    batch_size_inloop = batch_size
    max_iter_inloop = max_iter

    if print_output is True:
        print("Optimization starts with batch size {}".format(batch_size_inloop))

    while convergence == 0:
        while max_tol < tol and iteration < max_iter_inloop:
            # Using Pytorch's dataloader is slow. # TODO: Use Pytorch's dataloader in a future update.
            number_of_batches = int(np.ceil(x_train.shape[0] / batch_size_inloop))
            for i in range(number_of_batches):
                x_batch = x_train[i*batch_size_inloop:(i+1)*batch_size_inloop]
                y_batch = y_train[i*batch_size_inloop:(i+1)*batch_size_inloop]

                # Forward pass: compute predicted y by passing x to the model.
                y_batch_prediction = nn_model(x_batch)

                # Compute loss.
                loss = loss_fn(y_batch_prediction, y_batch)

                # Reset gradient
                optimizer.zero_grad()

                # Compute gradient (Backpropagation)
                loss.backward()

                # Take an optimization step
                optimizer.step()

            with th.no_grad():  # Close gradient calculation for cost calculation with new parameters.
                y_train_prediction = nn_model(x_train)  # Full sample prediction
                loss = loss_fn(y_train_prediction, y_train)  # Full sample loss
                loss_matrix[iteration] = loss
                if follow_cv is True:
                    y_cv_prediction = nn_model(x_cv)
                    loss_cv = loss_fn(y_cv_prediction, y_cv)  # CV sample loss
                    cv_loss_matrix[iteration] = loss_cv
                if iteration != 0:
                    tol = (loss_matrix[iteration] - loss_matrix[iteration-1]).abs_().item()
                    if follow_cv is True:
                        change_in_cv = (cv_loss_matrix[iteration] - cv_loss_matrix[iteration-1]).item()
                iteration = iteration + 1
                if iteration % 50 == 0 and print_output is True:
                    if follow_cv is True:
                        print("Epoch: {}/{}, Loss: {:.6e}, Tol: {:8e}, Change in CV Loss: {}".format(iteration,
                                                                                                     max_iter_inloop,
                                                                                                     loss.item(),
                                                                                                     tol,
                                                                                                     '+' if change_in_cv>= 0
                                                                                                     else '-'))
                    else:
                        print("Epoch: {}/{}, Loss: {:.6e}, Tol: {:.8e}".format(iteration, max_iter_inloop,
                                                                               loss.item(), tol))

        convergence = True if tol < max_tol else False

        if convergence is False:
            if int(batch_size_factor * batch_size_inloop) <= n_obs:
                batch_size_inloop = int(batch_size_factor * batch_size_inloop)
            else:
                if batch_size_inloop == n_obs:
                    raise ValueError("Convergence is not achieved with sample key {}.".format(sample_key))
                else:
                    batch_size_inloop = n_obs
            loss_matrix = th.cat((loss_matrix, th.zeros(max_iter, device=device)))
            if follow_cv is True:
                cv_loss_matrix = th.cat((cv_loss_matrix, th.zeros(max_iter, device=device)))
            max_iter_inloop = max_iter_inloop + max_iter
            if print_output is True:
                print("Batch size increased to {}".format(batch_size_inloop))
        else:
            if print_output is True:
                print("Convergence {}".format("achieved at:" if convergence is True else "failed."))
                print("Epoch: {}/{}, Loss: {:.6e}, Tol: {:.8e}".format(iteration, max_iter_inloop, loss.item(), tol))
    if follow_cv is False:
        y_cv_prediction = nn_model(x_cv)
        loss_cv = loss_fn(y_cv_prediction, y_cv)  # CV sample loss
        cv_loss_matrix = loss_cv  # Integer this time
    return nn_model, loss_matrix, cv_loss_matrix


def dn_estimation_with_cv(sample_key, n_input_layer, n_output_layer, activation_function, n_hidden_min, n_hidden_max,
                          loss_fn, idx_bootstrap, full_data, p_ident, e_ident, b_ident, d_ident=None, max_tol=1e-8,
                          max_iter=200, batch_size=32, batch_size_factor=2, device='cpu', print_output=True,
                          follow_cv=False, learning_rate=1e-4, cv_check_start=10):
    """
    Performs optimization with cross-validation. Creates an equally spaced integer array (round down) by taking
    n_hidden_min and n_hidden_max as inclusive borders. Performs optimization for each integer. Repeats the procedure
    by updating borders with number of hidden layers with the lowest cross-validation cost until the array is exhausted.
    :param sample_key: Integer. Which bootstrap sample to run estimation with?
    :param n_input_layer: Integer. Number of nodes at the input layer.
    :param n_output_layer: Integer. Number of nodes at the output layer.
    :param activation_function: th.nn item. Which non-linear activation function to use at the hidden layers.
    :param n_hidden_min: Integer or Tuple of Integers. Lower bound for number of node search for each hidden layer.
    Number of hidden layers is inferred from here and number of elements in this tuple (if tuple) must be the same as
    the number of elements in n_hidden_max.
    :param n_hidden_max: Integer or Tuple of Integers. Upper bound for number of node search for each hidden layer.
    Number of hidden layers is inferred from here and number of elements in this tuple (if tuple) must be the same as
    the number of elements in n_hidden_min.
    :param loss_fn: th.nn item. Which function to use to evaluate loss.
    :param idx_bootstrap: Dictionary of indices for training, cross-validation and test samples.
    :param full_data: Pandas DataFrame. Main data frame to draw bootstrap samples.
    :param p_ident: String. Unique starting strings of price columns.
    :param e_ident: String. Unique starting strings of expenditure columns.
    :param b_ident: String. Unique starting strings of budget share columns.
    :param d_ident: Optional String. Default: None. Unique starting strings of demographic columns.
    :param max_tol: Optional Float. Default: 1e-8. Maximum tolerance for convergence.
    :param max_iter: Optional Integer. Default: 200. Maximum number of epochs (iterations over all batches)
    before stopping optimization with the current batch size.
    :param batch_size: Optional Integer. Default: 32. Starting batch size for optimization.
    :param batch_size_factor: Optional Integer. Default: 2. Factor to increase batch size if converge fails. New batch
    size is batch_size_factor*batch_size if new batch size is less than the size of the training sample. If not, new
    batch size is set to the sample size.
    :param device: Optional String. Default: 'cpu'. Pass device information here if running the optimization on another
    device.
    :param print_output: Optional Boolean. Default: True. If optimization prints information.
    :param follow_cv: Optional Boolean. Default: False. If optimization tracks evaluation of cross-validation cost
    through optimization. If False, only the cross-validation cost after model training is recorded.
    :param learning_rate: Optional Float. Default: 1e-4. Learning rate for the optimizer.
    :param cv_check_start: Optional Integer. Default: 10. Cross-validation check is applied recursively starting from
     n_hidden_min and checking cross-validation cost for all possible number of nodes until n_hidden_min +
     cv_check_start. At this number of nodes, a line is fitted to measure the trend of cross-validation costs. If the
     trend is decreasing, cross-validation continues. If not, cross-validation stops. If the cross-validation continues,
     it repeats this tendency check with a moving window of last cv_check_start number of nodes until tendency turns
     positive or n_hidden_max is reached.
    :return: optimum_n_hidden, output_dict
    optimum_n_hidden: Integer. Optimum number of hidden layers.
    output_dict: Dictionary. Information on trained models during cross-validation. Keys of this dictionary are
    [n_hidden_layer', 'cv_loss', 'trained_model', 'loss_eval'], which reports number of hidden layers of an estimation,
    cross-validation loss of an estimation, trained model with that particular number of nodes, and evaluation of
    training loss during that particular optimization.
    """
    # Initialize dictionaries for storage
    output_dict = {'n_hidden_layer': [], 'cv_loss': [], 'trained_model': [], 'loss_eval': []}

    if n_hidden_min == n_hidden_max:
        raise ValueError("n_hidden_min and n_hidden_max are equal.")

    range_in = n_hidden_max - n_hidden_min
    n_hidden = np.linspace(n_hidden_min, n_hidden_max, range_in+1).astype(int)
    n_node = int(n_hidden[0])
    while n_node in n_hidden:
        nn_model = set_model(n_input_layer, n_node, n_output_layer, activation_function, device)
        optimizer = th.optim.Adam(nn_model.parameters(), learning_rate)
        if print_output is True:
            print("Estimation with hidden layer nodes: {}".format(n_node))
        trained_model, loss_eval, cv_loss_matrix = dn_estimation(sample_key, full_data, idx_bootstrap, p_ident,
                                                                 e_ident, b_ident, nn_model, optimizer, loss_fn,
                                                                 d_ident, max_tol, max_iter,
                                                                 batch_size, batch_size_factor, device,
                                                                 print_output, follow_cv)
        output_dict['n_hidden_layer'].append(n_node)
        output_dict['cv_loss'].append(cv_loss_matrix.item())
        output_dict['trained_model'].append(trained_model)
        output_dict['loss_eval'].append(loss_eval)

        if n_node == n_hidden[-1]:
            optimum_n_hidden = int(output_dict['n_hidden_layer'][int(np.argmin(output_dict['cv_loss']))])
            print("Cross-validation complete. Number of optimum hidden layer nodes: {}".format(optimum_n_hidden))
        elif np.where(n_hidden == n_node)[0].item() >= cv_check_start:
            x = np.arange(cv_check_start).reshape(cv_check_start, 1)
            y = np.array(output_dict['cv_loss'][-cv_check_start:]).reshape(cv_check_start, 1)
            if np.linalg.lstsq(x, y, rcond=None)[0].item() > 0:
                optimum_n_hidden = int(output_dict['n_hidden_layer'][int(np.argmin(output_dict['cv_loss']))])
                if print_output is True:
                    print("Last {} cross-validation losses have an increasing tendency. Cross-validation complete "
                          "with number of optimum hidden layer nodes: {}.".format(cv_check_start, optimum_n_hidden))
                break
            else:
                if print_output is True:
                    print("Last {} cross-validation losses have a decreasing tendency. Cross-validation continues.".format(cv_check_start))

        n_node = n_node+1

    return optimum_n_hidden, output_dict


def tdn_estimation(sample_key, full_data, idx_bootstrap, p_ident, e_ident, b_ident,
                   nn_model, optimizer, loss_fn, symmetry=False, negativity=False, lambdas=1., lambdas_factor=1.5,
                   d_ident=None, max_tol=1e-8,
                   max_iter=1000, batch_size=256, batch_size_factor=2, device='cpu', print_output=True):
    """
    Demand network optimizer.
    :param sample_key: Integer. Which bootstrap sample to use as the training sample.
    :param full_data: Pandas DataFrame. Main data frame to draw bootstrap samples.
    :param idx_bootstrap: Dictionary of indices for training, cross-validation and test samples.
    :param p_ident: String identifier for price columns. Starting unique pattern of the column names.
    :param e_ident: String identifier for expenditure columns. Starting unique pattern of the column name.
    :param b_ident: String identifier for budget share columns. Starting unique pattern of the column names.
    :param nn_model: Neural network model object. Obtained from set_model.
    :param optimizer: Neural network optimizer. Set with torch.optim.
    :param loss_fn: Loss function. Set with th.nn.
    :param symmetry: Optional Bool. Default: False. True if symmetry of the slutsky matrix is imposed.
    :param negativity: Optional Bool. Default: False. True if negativity of the slutsky matrix is imposed.
    :param theory_sample: Optional Pandas DataFrame or Float. Default:None. Data points at which symmetry and negativity
    are imposed. If not provided, and symmetry and/or negativity are imposed, full_data is used, i.e. constraints are
    imposed at all points. If provided, symmetry and/or negativity are imposed only at these points. If theory_sample is
    different than full_data, make sure to set batch_size to number of observations in full_data so each update step
    spans all observations. Otherwise, batches that are fed into an optimization step may not contain some observations
    for which the optimization is penalized. If float, theory_sample must be between 0 and 1. In this case, at each
    step, the optimizer selects batch_size*theory_sample number of random observations to impose restrictions. This is
    equal to imposing restrictions at all observations but should perform faster in some cases.
    :param d_ident: Optional String. Default: None. String identifier for demographics columns. Starting unique pattern
    of the column names.
    :param max_tol: Optional Float. Default: 1e-8. Maximum tolerance for convergence.
    :param max_iter: Optional Integer. Default: 200. Maximum number of iterations before increase in batch-size.
    :param batch_size: Optional Integer. Default: 32. Starting batch size. Capped at number of observations.
    :param batch_size_factor: Optional Integer. Default: 2. Factor of increase in batch_size if convergence is not
    achieved. New batch_size is batch_size*batch_size_factor if sample size is not exhausted. Batch size is capped at
    number of observations.
    :param device: Optional String. Default: 'cpu'. Which device to use for optimization.
    :param print_output: Optional Bool. Default: True. Should optimization print information during epochs?
    :param follow_cv: Optional Bool. Default: True. Should optimization follow evolution of cross-validation cost during
    epochs? If set to False, cross-validation cost after convergence is reported.
    :return: nn_model, loss_matrix, cv_loss.
    nn_model: Trained version of the input nn_model.
    loss_matrix: Array of training loss after each iteration.
    cv_loss: If follow_cv is True, an array of cross-validation loss after each iteration. Else, a float that reports
    cross-validation cost of trained model.
    """
    # Obtain bootstrap sample
    training_data, cv_data = pull_bs_sample(sample_key, idx_bootstrap, full_data, p_ident, e_ident, b_ident, d_ident)
    x_train = training_data.x
    y_train = training_data.y
    x_cv = cv_data.x
    y_cv = cv_data.y

    if device != 'cpu':
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        x_cv = x_cv.to(device)
        y_cv = y_cv.to(device)

    penalty_approach = True if symmetry is True or negativity is True else False

    # Optimization
    if print_output is True:
        print("Optimization starts with batch size {}".format(batch_size))
    nn_model, train_loss_matrix, convergence = dn_optimizer(x_train, y_train, nn_model, loss_fn, optimizer,
                                                            batch_size, batch_size_factor, max_tol, max_iter,
                                                            symmetry, negativity, lambdas,
                                                            device, print_output)
    if penalty_approach is True:
        train_loss_dict = {str(lambdas): train_loss_matrix}
        iteration = 0
        while iteration <= max_iter:
            iteration = iteration + 1
            lambdas = lambdas * lambdas_factor
            if print_output is True:
                print("Penalty approach starts with lambdas={.2e}".format(lambdas))
            nn_model, train_loss_matrix, convergence = dn_optimizer(x_train, y_train, nn_model, loss_fn, optimizer,
                                                                    batch_size, batch_size_factor, max_tol, max_iter,
                                                                    symmetry, negativity, lambdas,
                                                                    device, print_output)
            train_loss_dict[str(lambdas)] = train_loss_matrix
            penalty_tol = train_loss_matrix[str(lambdas/lambdas_factor)][-1] - train_loss_dict[str(lambdas)][-1]
            penalty_tol = penalty_tol.abs_().item()
            if penalty_tol <= max_tol:  # Convergence achieved.
                if print_output is True:
                    print("Penalty approach convergence achieved with penalty coefficient {.2e}.".format(lambdas))
                break
            else:
                if print_output is True:
                    print("Penalty approach continues with penalty coefficient {.2e}.".format(lambdas))

    # Calculate cross-validation cost
    y_cv_prediction = nn_model(x_cv)
    loss_cv = loss_fn(y_cv_prediction, y_cv)  # CV sample loss
    cv_loss = loss_cv

    if penalty_approach is False:
        return nn_model, train_loss_matrix, cv_loss, convergence
    else:
        return nn_model, train_loss_dict, cv_loss, convergence


def epoch_dn(loss_fn, x_batch, y_batch, nn_model, n_good,
             optimizer, symmetry=True, negativity=True, lambdas=1.):
    """
    This function takes an optimizer, provides optimizer.step() given inputs, and outputs updated optimizer.
    :param loss_fn: Loss function. Set with th.nn.
    :param x_batch: n_batch-by-n_input input tensor.
    :param y_batch: n_batch-by-n_good output tensor.
    :param nn_model: Neural network model object. Obtained from set_model.
    :param n_good: Integer. Number of goods: n_good.
    :param optimizer: Neural network optimizer. Set with torch.optim.
    :param symmetry: Optional Boolean. Default: True. Should symmetry be imposed?
    :param negativity: Optional Boolean. Default: True. Should negativity be imposed?
    :param lambdas: Optional Float. Default: 1.. Coefficient of theoretical loss.
    :return: optimizer: A step updated optimizer.
    """
    # Forward pass: compute predicted y by passing x to the model.
    y_batch_prediction = nn_model(x_batch)

    # Compute loss.
    loss = loss_fn(y_batch_prediction, y_batch)
    if symmetry is True or negativity is True:
        x_batch_theory_cost_input = x_batch.clone().detach()
        income_e, unc_pr_el, comp_pr_el, slutsky_matrix = dn_elasticities(x_batch_theory_cost_input,
                                                                          n_good, nn_model)  # Elasticities
        symmetry_cost, negativity_cost = theoretical_cost(y_batch_prediction, slutsky_matrix,  # Costs of violations
                                                          symmetry=symmetry, negativity=negativity)
        theory_loss = symmetry_cost + negativity_cost  # Compute total cost of violations
        loss = loss + lambdas*theory_loss  # Compute total loss

    # Reset gradient
    optimizer.zero_grad()  # Restart accumulating gradients.

    # Compute gradient
    loss.backward()  # Backprop

    # Take an optimization step
    optimizer.step()  # Take an optimization step

    return optimizer


def dn_optimizer(x_train, y_train, nn_model, loss_fn, optimizer,
                 batch_size=256, batch_size_factor=2, max_tol=1e-8, max_iter=1000,
                 symmetry=False, negativity=False, lambdas=1.,
                 device='cpu', print_output=True):
    n_obs, n_good = y_train.shape
    convergence = False
    batch_size_inloop = batch_size
    max_iter_inloop = max_iter
    iteration = 0
    tol = 1e5
    train_loss_matrix = th.zeros([max_iter], device=device)
    x_theory_cost_input = x_train.clone()  # To keep gradients out of x_train

    # Main Optimization Loop
    while convergence is False:
        number_of_batches = int(np.ceil(x_train.shape[0] / batch_size_inloop))
        while max_tol < tol and iteration < max_iter_inloop:
            # Pytorch(0.41)'s dataloader is slow here. So I proceed with manual batching.
            for i in range(number_of_batches):
                x_batch = x_train[i*batch_size_inloop:(i+1)*batch_size_inloop]
                y_batch = y_train[i*batch_size_inloop:(i+1)*batch_size_inloop]
                optimizer = epoch_dn(loss_fn, x_batch, y_batch, nn_model, n_good,
                                     optimizer, symmetry=symmetry, negativity=negativity)
            y_train_prediction = nn_model(x_train)  # Full sample prediction
            train_loss = loss_fn(y_train_prediction, y_train)  # Full sample loss
            if symmetry is True or negativity is True:
                x_theory_cost_input.grad = th.zeros_like(x_theory_cost_input)
                income_e, unc_pr_el, comp_pr_el, slutsky_matrix = dn_elasticities(x_theory_cost_input, n_good, nn_model)
                symmetry_cost, negativity_cost = theoretical_cost(y_train_prediction, slutsky_matrix,
                                                                  symmetry=symmetry, negativity=negativity)
                train_theory_loss = symmetry_cost + negativity_cost  # Compute loss
                train_loss = train_loss + lambdas*train_theory_loss
            train_loss_matrix[iteration] = train_loss
            if iteration != 0:
                tol = (train_loss_matrix[iteration] - train_loss_matrix[iteration-1]).abs_().item()
            iteration = iteration + 1
            if iteration % 1 == 0 and print_output is True:
                print("Epoch: {}/{}, Loss: {:.6e}, Tol: {:.8e}".format(iteration, max_iter_inloop,
                                                                       train_loss.item(), tol))

        convergence = True if tol < max_tol else False

        if convergence is False:
           # if batch_size_inloop == n_obs:
                #break
            if int(batch_size_factor * batch_size_inloop) <= n_obs:
                batch_size_inloop = int(batch_size_factor * batch_size_inloop)
            else:
                batch_size_inloop = n_obs
            train_loss_matrix = th.cat((train_loss_matrix, th.zeros(max_iter, device=device)))
            max_iter_inloop = max_iter_inloop + max_iter
            if print_output is True:
                print("Batch size increased to {}".format(batch_size_inloop))
        else:
            if print_output is True:
                print("Convergence {}".format("achieved at:" if convergence is True else "failed."))
                print("Epoch: {}/{}, Loss: {:.6e}, Tol: {:.8e}".format(iteration, max_iter_inloop, train_loss.item(),
                                                                       tol))

    return nn_model, train_loss_matrix, convergence


