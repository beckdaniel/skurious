
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


def train_model(X_train, y_train, params):
    """Train an Active Learner model. Is is actually a GP with an
    additional query method. GP method calls are transparently passed
    to the ActiveLearner object.
    """
    #############################
    # FEATURE SELECTION CODE (MAYBE?) HERE
    #
    # I think the "X_train" parameter should always
    # have the full feature set (90ish?). Then, before
    # training the model, you "filter" the X_train
    # according to the current "optimal" feature set,
    # selecting the right columns using some kind of
    # "mask". What I mean by a mask is just an array with
    # the column numbers like:
    # feat_mask = [1,4,8,19,56]
    #############################

    kernel = GPy.kern.rbf(D=17, ARD=True)
    al = GPActiveLearner(GPy.models.GP_regression(X_train, y_train, kernel))

    # Something like that:
    # X_train_selected = X_train[feat_mask]
    # kernel = GPy.kern.rbf(D=X_train_selected.shape[0], ARD=True)
    # al = GPActiveLearner(GPy.models.GP_regression(X_train_selected, y_train, kernel))

    al.constrain_positive('')
    old_params = al._get_params()
    if X_train.shape[0] % 100 == 0: #WARNING: must be multiple of START_SAMPLES
        try:
            al.optimize() # We could use optimize_restarts instead
            old_params = al._get_params()
            params = old_params
        except (ValueError, np.linalg.linalg.LinAlgError) as e:
            # Sometimes the optimization fails because of errors in
            # the matrix... I have no idea why this happens...
            # For now, when this happens, I fallback to the previous
            # parameters.
            print "WARNING: exception here: %s" % e
            print "Aborting optimization..."
            al._set_params(old_params)
    else:
        al._set_params(params)

    # Here you somehow do your feature selection stuff and update
    # the "feat_mask" variable
    # <feature_selection_code>
    # return al, feat_mask
    return al


def run_experiment(strategy, scaler, X_train, y_train, X_query,
                   y_query, output_dir, X_test, y_test, id_exp=1):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    extra = create_extra(strategy, X_query, id_exp)
    results_file = os.path.join(output_dir, 'results.tsv')
    queries_file = os.path.join(output_dir, 'queries.tsv')

    with open(results_file, 'w') as r, open(queries_file, 'w') as q:
        results = csv.writer(r, delimiter='\t')
        queries = csv.writer(q, delimiter='\t')
        params = []

        while X_query.shape[0] > 0:
            # This loops iterates on the query pool, adding
            # a new instance to X_train and y_train every
            # iteration.

            # train model
            al = train_model(X_train, y_train, params)
            # al, feat_mask = train_model(X_train, y_train, params)
            params = al._get_params()
            
            #####################
            # FEAT SELECTION AGAIN HERE
            #
            # Since "X_query" will always have the full
            # feature set, you should selected the feats
            # using the "feat_mask" again
            #
            # X_query_selected = X_query[feat_mask]
            # best_i = al.argquery(X_query_selected, strategy, extra)
            ####################
            best_i = al.argquery(X_query, strategy, extra)
            best_X = X_query[best_i]
            best_y = y_query[best_i]

            # update pool
            X_train = np.concatenate((X_train, [best_X]))
            y_train = np.concatenate((y_train, [best_y]))
            X_query = np.delete(X_query, best_i, axis=0)
            y_query = np.delete(y_query, best_i, axis=0)
            if strategy == "id" or strategy == "density":
                extra["similarities"] = np.delete(extra["similarities"], best_i, axis=0)
                extra["similarities"] = np.delete(extra["similarities"], best_i, axis=1)
                
            # report results
            queries.writerow(np.concatenate((scaler.inverse_transform(best_X),
                                             best_y)))
            preds = al.predict(X_test)[0]
            mae = MAE(preds, y_test)
            results.writerow([X_train.shape[0], mae])
            print "QUERIES: %d\tMAE: %.5f" % (X_train.shape[0], mae)
            



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
Y_TEST = sys.argv[4]
OUTPUT_DIR = sys.argv[5]
START_SAMPLES = 100

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
y_test = np.array([[a] for a in np.loadtxt(Y_TEST)])

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
               y_query, OUTPUT_DIR, X_test, y_test)

