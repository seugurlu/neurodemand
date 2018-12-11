import tensorflow as tf
import pandas as pd


class Cdn:
    def __init__(self, sample_key, cdn_info):
        self.sample_key = 0
        self.data_path = cdn_info['data_path']
        self.index_col = cdn_info['index_col']
        self.p_ident = cdn_info['p_ident']
        self.e_ident = cdn_info['e_ident']
        self.d_ident = cdn_info['d_ident']
        self.idx_training = cdn_info['idx_training']
        self.idx_cv = cdn_info['idx_cv']
        self.idx_test = cdn_info['idx_test']

        # To be assigned when proper functions are initialized
        self.train_x, self.train_y = [], []
        self.cv_x, self.cv_y = [], []
        self.test_x, self.test_y = [], []
        self.n_goods = []

    def gen_input_data(self):
        """
        Returns inputs and outputs from a dataset.
        """

        # Extract data to work with using the provided indices.
        data = pd.read_csv(self.data_path, index_col=self.index_col)
        if self.idx is None:
            current_data = data
        else:
            current_data = data.loc[self.idx, :]

        # Extract input and output information: prices, expenditures, demographics as covariates and budget shares
        # as explanatory variables.
        prices = current_data.loc[:, current_data.columns.str.startswith(self.p_ident)]  # Init with prices
        expenditures = current_data.loc[:, current_data.columns.str.startswith(self.e_ident)]  # Expenditures
        features = pd.concat([prices, expenditures], axis=1, sort=False)
        if self.d_ident is not None:
            demographics = current_data.loc[:, current_data.columns.str.startswith(self.d_ident)]  # Demographics
            features = pd.concat([features, demographics], axis=1, sort=False)

        outputs = current_data.loc[:, current_data.columns.str.startswith(self.b_ident)]  # Budget shares

        return features, outputs



def cdn(sample_key, cdn_info):
    # session_conf = tf.ConfigProto(
    #     intra_op_parallelism_threads=1,
    #     inter_op_parallelism_threads=1)
    # sess = tf.InteractiveSession(config=session_conf)

    """Input Data"""
    x_train, y_train = gen_input_data()

    """Edit Input Output Data"""
    # x_train = np.log( x_train )#Generate input matrix as homogeneity adjusted log
    # x_cv = np.log( x_cv )#Generate input matrix as homogeneity adjusted log
    # x_test = np.log( x_test )#Generate input matrix as homogeneity adjusted log
    idx_training = idx_bootstrap[sample_key]['training_sample']
    idx_cv = idx_bootstrap[sample_key]['cv_sample']
    idx_test = idx_bootstrap[sample_key]['test_sample']

    x_train = full_data.iloc[idx_training, :number_goods + 1]
    x_cv = full_data.iloc[idx_cv, :number_goods + 1]
    x_test = full_data.iloc[idx_test, :number_goods + 1]
    y_train = full_data.iloc[idx_training, number_goods + 4:]
    y_cv = full_data.iloc[idx_cv, number_goods + 4:]
    y_test = full_data.iloc[idx_test, number_goods + 4:]

    x_size = number_goods + 1  # Size of the input layer: + expenditure (1)
    y_size = number_goods  # Size of the output layer

    '''Initialize Parameters and arrays'''
    hh_start = 2  # Set minimum number of hidden layers
    hh_size = hh_start  # Initialize
    window = 5
    hh_end = hh_start + window
    hh_max = 2 * number_goods + window
    train_results = pd.DataFrame(columns=['hh_size', 'train_acc'])
    cv_results = pd.DataFrame(columns=['hh_size', 'cv_acc'])
    test_results = pd.DataFrame(columns=['hh_size', 'test_acc'])

    '''Set Placeholders & Constants for NN'''
    x = tf.placeholder(tf.float32, shape=[None, x_size])
    y = tf.placeholder(tf.float32, shape=[None, y_size])
    length = tf.placeholder(tf.int32, shape=())

    idx = -1  # Initialize
    while hh_size <= hh_end:
        '''This is the main optimization loop'''
        '''Variable Stopping Criteria for hh_size'''
        idx = idx + 1
        train_results.loc[idx] = [hh_size, 0]
        cv_results.loc[idx] = [hh_size, 0]
        test_results.loc[idx] = [hh_size, 0]

        '''Variables'''
        w_1 = tf.Variable(tf.truncated_normal([x_size, hh_size], stddev=0.1), dtype=tf.float32)
        w_2 = tf.Variable(tf.truncated_normal([hh_size, y_size], stddev=0.1), dtype=tf.float32)
        b_1 = tf.Variable(tf.random_normal([hh_size]), dtype=tf.float32)
        b_2 = tf.Variable(tf.random_normal([y_size]), dtype=tf.float32)

        '''Model'''
        yhat, z1 = nf.forwardprop(x, w_1, w_2, b_1, b_2)
        train_cost = tf.reduce_mean((y - yhat) ** 2)
        train_step = tf.train.AdamOptimizer(epsilon=1e-4).minimize(train_cost)
        accuracy_cost = tf.reduce_mean((y - yhat) ** 2)  # To evaluate accuracy

        '''Estimation'''
        tf.global_variables_initializer().run()  # Initialize variables
        train_accuracy_post_optimization = 10.0  # initialize stopping criterion

        '''Updates start here'''
        # Stochastic Adam
        print("Stochastic gradient starts with hhsize: %s" % (hh_size))
        for epoch in range(stochastic_grad_epoch_limit):
            train_accuracy_pre_optimization = train_accuracy_post_optimization
            # Start with stochastic to converge around minimum faster
            for i in range(len(x_train)):
                train_step.run({x: x_train[i:i + 1], y: y_train[i:i + 1], length: 1})
            train_accuracy_post_optimization = train_cost.eval({x: x_train, y: y_train, length: len(x_train)})
            print('Stochastic gradient continues at hhsize: %s, epoch %s. Post_cost: %.8f' % (
            hh_size, epoch, train_accuracy_post_optimization))
            if tf.abs(train_accuracy_pre_optimization - train_accuracy_post_optimization).eval() <= tol_stochastic:
                print("Stochastic gradient converged")
                break

        print("Minibatch gradient starts with hhsize: %s" % (hh_size))
        for epoch in range(mini_batch_grad_epoch_limit):
            train_accuracy_pre_optimization = train_accuracy_post_optimization
            # Continue with mini batch
            for i in range(int(len(x_train) / batch_size)):  # Train with each example
                batch_x = x_train[i * batch_size:(i + 1) * batch_size]
                batch_y = y_train[i * batch_size:(i + 1) * batch_size]
                train_step.run({x: batch_x, y: batch_y, length: batch_size})
            remaining_observations = len(x_train) % batch_size
            if remaining_observations != 0:
                batch_x = x_train[len(x_train) - remaining_observations:len(x_train)]
                batch_y = y_train[len(y_train) - remaining_observations:len(y_train)]
                train_step.run({x: batch_x, y: batch_y, length: remaining_observations})
            train_accuracy_post_optimization = train_cost.eval({x: x_train, y: y_train, length: len(x_train)})
            print('Minibatch gradient continues at hhsize: %s, epoch %s. Post_cost: %.8f' % (
            hh_size, epoch, train_accuracy_post_optimization))
            if tf.abs(train_accuracy_pre_optimization - train_accuracy_post_optimization).eval() <= tol_iter:
                print("Stochastic gradient converged")
                break

        train_accuracy = accuracy_cost.eval({x: x_train, y: y_train, length: len(x_train)})
        cv_accuracy = accuracy_cost.eval({x: x_cv, y: y_cv, length: len(x_cv)})
        test_accuracy = accuracy_cost.eval({x: x_test, y: y_test, length: len(x_test)})

        train_results.loc[idx, 'train_acc'] = train_accuracy
        cv_results.loc[idx, 'cv_acc'] = cv_accuracy
        test_results.loc[idx, 'test_acc'] = test_accuracy

        # save_name = "./output/Temp/nfnn/coefficients_nfnn_"+str(sample_key)+"_"+str(hh_size)
        # np.savez(save_name, w_1 = w_1.eval(), w_2 = w_2.eval(), b_1 = b_1.eval(), b_2 = b_2.eval(), hh_size = hh_size,
        # train_accuracy = train_accuracy, cv_accuracy = cv_accuracy, test_accuracy = test_accuracy )

        # Stop for while
        if hh_size == hh_end:
            reg = linear_model.LinearRegression()
            x_reg = cv_results.loc[:, 'hh_size'].tail(window).values.reshape(window, 1)
            y_reg = cv_results.loc[:, 'cv_acc'].tail(window).values.reshape(window, 1)
            reg.fit(x_reg, y_reg)
            if reg.coef_ < -tol_iter:
                if hh_end <= hh_max:
                    hh_end += 1
                else:
                    hh_end = hh_end

        hh_size += 1

    hh_size = int(
        cv_results.loc[cv_results['cv_acc'].idxmin(), 'hh_size'])  # Size with the best cross-validation result
    with np.load("./output/Temp/nfnn/coefficients_nfnn_" + str(sample_key) + "_" + str(hh_size) + '.npz') as data:
        w_1 = data['w_1']
        w_2 = data['w_2']
        b_1 = data['b_1']
        b_2 = data['b_2']
        hh_size = data['hh_size']
        train_accuracy = data['train_accuracy']
        cv_accuracy = data['cv_accuracy']
        test_accuracy = data['test_accuracy']
    return (train_accuracy, cv_accuracy, test_accuracy, w_1, w_2, b_1, b_2)
    raw_data.close()
    sess.close()
