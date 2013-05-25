
import numpy as np
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics.pairwise import cosine_similarity as CS
from sklearn import preprocessing as pp
from skurious.skurious import GPActiveLearner
import sys
import os
import csv
import GPy


def create_extra(strategy, X_query, id_exp):
    """ Instantiate an "extra" dict, used in the query method."""
    extra = {}
    if strategy == "id" or strategy == "density":
        extra["similarities"] = CS(X_query, X_query)
        extra["id_exp"] = id_exp
    return extra


def opt_feat(m):
    """ Optimize a model and do feat selection.
    """
    # We force an isotropic kernel first to get initial values
    m.tie_params('lengthscale')
    m.constrain_positive('')
    m.optimize(max_f_eval=500, messages=True)

    # Now we resume ARD
    m.untie_everything()
    m.optimize(optimizer='lbfgsb', max_f_eval=1000, messages=True) 
    
    # Feat select
    lengthscales = m.get('lengthscale')
    sorted_ls = np.argsort(lengthscales)
    feat_mask = sorted_ls[:10]
    #print lengthscales
    #print sorted_ls
    #print feat_mask
    return m, feat_mask


def train_model(X_train, y_train):
    """ Train a GP.
    """
    D = X_train.shape[1]
    rbf = GPy.kern.rbf(D=D, ARD=True)
    noise = GPy.kern.white(D)
    kernel = rbf + noise
    al = GPActiveLearner(GPy.models.GP_regression(X_train, y_train, kernel))
    return al


def run_experiment(strategy, scaler, X_train, y_train, X_query,
                   y_query, output_dir, X_test, id_exp=1):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    results_dir = os.path.join(output_dir,'results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    extra = create_extra(strategy, X_query, id_exp)
    queries_file = os.path.join(output_dir, 'queries.tsv')
    
    with open(queries_file, 'w') as q:
        # write the first instances in the queries file
        queries = csv.writer(q, delimiter='\t')
        for i, instance in enumerate(X_train):
            queries.writerow(np.concatenate((scaler.inverse_transform(instance),
                                             y_train[i])))

        # Perform first optimization and feat select step
        al = train_model(X_train, y_train)
        al, feat_mask = opt_feat(al)
        lscales = al.get('lengthscale')[feat_mask]
        vars = al.get('variance')
        print al.estimator

        # Initial rate is 20
        rate = 20
        
        while X_query.shape[0] > 0:
            # First, we train a new model with the current features
            X_train_sel = X_train[:, feat_mask]
            al = train_model(X_train_sel, y_train)
            al.set('lengthscale', lscales)
            al.set('variance', vars)
            #print feat_mask
            #print al.estimator
            
            # Second, we report results on the test set
            results_file = os.path.join(results_dir, str(X_train.shape[0]) + '.tsv')
            X_test_sel = X_test[:, feat_mask]
            with open(results_file, 'w') as r:
                results = csv.writer(r, delimiter='\t')                
                preds = al.predict(X_test_sel)[0]
                for pred in preds:
                    results.writerow(pred)
                print "QUERIES: %d" % X_train.shape[0]

            # Third, we query a new instance
            X_query_sel = X_query[:, feat_mask]
            best_i = al.argquery(X_query_sel, strategy, extra)
            best_X = X_query[best_i]
            best_y = y_query[best_i]
                
            # Fourth, we update training and query sets
            X_train = np.concatenate((X_train, [best_X]))
            y_train = np.concatenate((y_train, [best_y]))
            X_query = np.delete(X_query, best_i, axis=0)
            y_query = np.delete(y_query, best_i, axis=0)
            if strategy == "id" or strategy == "density":
                extra["similarities"] = np.delete(extra["similarities"], best_i, axis=0)
                extra["similarities"] = np.delete(extra["similarities"], best_i, axis=1)

            # Fifth, we check if its time to do a new opt-feat step
            if X_train.shape[0] % rate == 0:
                al = train_model(X_train, y_train)
                al, feat_mask = opt_feat(al)
                lscales = al.get('lengthscale')[feat_mask]
                vars = al.get('variance')
                print al.estimator
                print feat_mask
                rate *= 2


########################
# Read command line args
# X_TRAIN = features for training set
# Y_TRAIN = score for training set (HTER)
# X_TEST = features for test set
# Y_TEST = score for test set (HTER)
# OUTPUT_DIR = directory to save results, it will save two files:
#  <strategy>/<number>.tsv = MAE scores on test set
#  <strategy_queries>/<number>.tsv = contains the training set, but in query order.
# START_SAMPLES = number of instances used to train the first model.
######################
X_TRAIN = sys.argv[1]
Y_TRAIN = sys.argv[2]
X_TEST = sys.argv[3]
OUTPUT_DIR = sys.argv[4]
START_SAMPLES = int(sys.argv[5])

###################
# This is just to suppress scientific notation when printing values on screen
###################
np.set_printoptions(suppress=True)

###################
# This is where data is actually read.
# I assume tab-separated values.
# The "y" arrays must be converted to matrices because GPy requires them like that.
###################
X_train = np.loadtxt(X_TRAIN)
X_test = np.loadtxt(X_TEST)
y_train = np.array([[a] for a in np.loadtxt(Y_TRAIN)])

##################
# I'm applying scaling on the full training set before splitting
# into training and query set. You could also use GPy to scale them
# when training each model but results will be different (since the scaling
# will be different for each model). Maybe using GPy to do scaling is
# a better choice than do it here?
#################
scaler = pp.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#################
# This is where I split the training set into
# a "start training set" and the "query set".
#################
X_query = X_train[START_SAMPLES:]
y_query = y_train[START_SAMPLES:]
X_train = X_train[:START_SAMPLES]
y_train = y_train[:START_SAMPLES]

###############
# This launches the experiment.
# The first parameter is the AL strategy, it can be:
#
# "random": passive learning, no AL
# "us": uses the variance to select new instances
# "id": leverages between variance and a density measure,
#       needs to set an additional "id_exp" parameter (default=1)
#
# From my experiments here, I would start with the "us" strategy.
# We should probably do experiments with the "random" strategy too,
# as a baseline.
#
# The experiment will save MAE scores on a test set and also save
# the selected instances.
##############
run_experiment("us", scaler, X_train, y_train, X_query,
               y_query, OUTPUT_DIR, X_test)

