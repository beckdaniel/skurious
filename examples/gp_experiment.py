
import numpy as np
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics.pairwise import cosine_similarity as CS
from sklearn import preprocessing as pp
from skurious.skurious import GPActiveLearner
from skurious.ensemble import Bagging
import sys
import os
import csv
import GPy

def run_experiment(strategy, scaler, base_X_train, base_y_train, base_X_query,
                   base_y_query, output_dir, X_test, y_test, id_exp=1):



    outdir = os.path.join(OUTPUT_DIR, strategy)
    qdir = os.path.join(OUTPUT_DIR, strategy + "_queries")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if not os.path.exists(qdir):
        os.mkdir(qdir)

    for i in range(5):
        X_query = np.copy(base_X_query)
        y_query = np.copy(base_y_query)
        X_train = np.copy(base_X_train)
        y_train = np.copy(base_y_train)
        extra = {}
        if strategy.startswith("id"):
            extra["densities"] = (CS(X_query, X_query) ** id_exp)
            strategy = "id"

        results_file = os.path.join(outdir, str(i)+'.tsv')
        queries_file = os.path.join(qdir, str(i)+'.tsv')
        with open(results_file, 'w') as r, open(queries_file, 'w') as q:
            results = csv.writer(r, delimiter='\t')
            queries = csv.writer(q, delimiter='\t')
            while X_query.shape[0] > 0:
            
                # obtain query
                kernel = GPy.kern.rbf(D=17, ARD=True)
                al = GPActiveLearner(GPy.models.GP_regression(X_train, y_train, kernel))
                al.constrain_positive('')
                try:
                    #al.optimize_restarts(Nrestarts=10)
                    al.optimize()
                    best_i = al.argquery(X_query, strategy, extra)
                except (ValueError, np.linalg.linalg.LinAlgError) as e:
                    print "WARNING: exception here: %s" % e
                    print "Choosing a random instance from query pool"
                    best_i = np.random.randint(0, high=X_query.shape[0])
                    kernel = GPy.kern.rbf(D=17, ARD=True)
                    al = GPActiveLearner(GPy.models.GP_regression(X_train, y_train, kernel))
                #al.fit(X_train, y_train)

                best_X = X_query[best_i]
                best_y = y_query[best_i]

                # update pool
                X_train = np.concatenate((X_train, [best_X]))
                y_train = np.concatenate((y_train, [best_y]))
                X_query = np.delete(X_query, best_i, axis=0)
                y_query = np.delete(y_query, best_i, axis=0)
                if strategy == "id":
                    extra["densities"] = np.delete(extra["densities"], best_i, axis=0)
                    extra["densities"] = np.delete(extra["densities"], best_i, axis=1)
            
                # report results
                queries.writerow(np.concatenate((scaler.inverse_transform(best_X),
                                                 best_y)))
                #cl = GPy.models.GP_regression(X_train, y_train, kernel)
                #m.constrain_positive('')
                #m.optimize()
                preds = al.predict(X_test)[0]
                mae = MAE(preds, y_test)
                results.writerow([X_train.shape[0], mae])
                print "QUERIES: %d\tMAE: %.5f" % (X_train.shape[0], mae)



X_TRAIN = sys.argv[1]
Y_TRAIN = sys.argv[2]
X_TEST = sys.argv[3]
Y_TEST = sys.argv[4]
OUTPUT_DIR = sys.argv[5]
START_SAMPLES = 100

np.set_printoptions(suppress=True)

X_train = np.loadtxt(X_TRAIN)
X_test = np.loadtxt(X_TEST)
y_train = np.array([[a] for a in np.loadtxt(Y_TRAIN)])
y_test = np.array([[a] for a in np.loadtxt(Y_TEST)])

scaler = pp.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

base_X_query = X_train[START_SAMPLES:]
base_y_query = y_train[START_SAMPLES:]
base_X_train = X_train[:START_SAMPLES]
base_y_train = y_train[:START_SAMPLES]

#kernel = GPy.kern.rbf(D=17, ARD=True)
#m = GPy.models.GP_regression(X_train, y_train, kernel)
#m.constrain_positive('')
#m.optimize()
#print m
#print MAE(m.predict(X_test)[0], y_test)


run_experiment("id_minus_10", scaler, base_X_train, base_y_train, base_X_query,
               base_y_query, OUTPUT_DIR, X_test, y_test, id_exp=-10)
run_experiment("id_1", scaler, base_X_train, base_y_train, base_X_query,
               base_y_query, OUTPUT_DIR, X_test, y_test)
run_experiment("random", scaler, base_X_train, base_y_train, base_X_query,
               base_y_query, OUTPUT_DIR, X_test, y_test)
run_experiment("us", scaler, base_X_train, base_y_train, base_X_query,
               base_y_query, OUTPUT_DIR, X_test, y_test)
