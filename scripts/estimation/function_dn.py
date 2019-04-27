"""DN estimation function"""


def cross_validation(sample_key, ):  # Define optimization algorithm with cross-validation
    from tensorflow import keras  # Import keras for each thread
    keras.backend.set_session(tf.Session(config=config))  # Set backend with config options
    optimizer = keras.optimizers.Adam(lr=learning_rate, epsilon=epsilon)  # Set optimizer
    # Pick training and cross-validation data for this particular bootstrap
    idx_training = idx_bootstrap[sample_key]['training_sample']  # Retrieve training set
    idx_cv = idx_bootstrap[sample_key]['cv_sample']  # Retrieve cross-validation set
    x_train, y_train = nf.prepare_data(full_data, p_ident, e_ident, b_ident,
                                       d_ident=d_ident, idx=idx_training)  # Retrieve input and output for training
    x_cv, y_cv = nf.prepare_data(full_data, p_ident, e_ident, b_ident,
                                 d_ident=d_ident, idx=idx_cv)  # Retrieve input and output for cross-validation

    # Estimation
    def estimation(n_node):  # Define optimization routine
        file_path = "./output/temp/dn/cross_validation/sample_{}_node_{}.h5".format(sample_key, n_node)  # Save path
        dn_model = nf.build_model(n_node, n_goods, optimizer, loss_fn, activation_fn=activation_function)  # Set model
        callbacks = [keras.callbacks.EarlyStopping('loss', min_delta=mini_batch_tol)]  # Define callbacks
        history = dn_model.fit(x=x_train, y=y_train, batch_size=mini_batch_training_batch_size,  #Optimization
                               epochs=mini_batch_training_epoch_limit, callbacks=callbacks, verbose=0,
                               validation_data=(x_cv, y_cv))
        dn_model.save(file_path)  # Save optimization output
        return history.history  # Return output
    cv_results = {'Number of Nodes': [], 'Loss History': []}  # Pre-allocate result dictionary
    for node in n_hidden_node_search_set:  # Loop over potential number of nodes
        cv_results['Number of Nodes'].append(node)
        cv_results['Loss History'].append(estimation(node))
    keras.backend.clear_session()  # Reset session
    return cv_results
