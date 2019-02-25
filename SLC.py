# Create a new classifier which is based on the sckit-learn BaseEstimator and ClassifierMixin classes

# import necessary modules

from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
import numpy as np
from seaborn import heatmap

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import clone
from sklearn.utils import resample
from sklearn.metrics import accuracy_score

class SuperLearnerClassifier(BaseEstimator, ClassifierMixin):
    
    """An ensemble classifier that uses heterogeneous models at the base layer and an aggregation model at 
    the aggregation layer.  A k-fold cross validation is used to generate training data for the stack layer model.

    Parameters
    ----------
        
    stack_estimator: string, optional, (default = 'dtc')
        The estimator used at the stack layer.  Options are 'dtc' (Decision Tree classifier), 'rfc' (Random Forest classifier), 
        'logr' (Logistic Regression), 'knn' (k-Nearest Neighbours classifier), 'mlpc' (Multi-layer Perceptron classifier), 
        'rsvc' (C-Support Vector Classification with rbf kernel), and 'sgdc' (Stochastic Gradient Descent classifier 
        with modified huber loss function)
        
    base_estimator_set: string representing list, optional, (default = 'default_set')
        A list of base estimators used to train the stack layer.  Available options are:
        all_clfs_: [dtc, rfc, logr, knn, mlpc, rsvc, sgdc] 
            - All 7 available estimators.
        default_set_: [rfc, logr, knn, mlpc, rsvc, sgdc] 
            - set of 6 estimators as required for assignment default; dtc omitted as it is used at stack layer by default.
        speedy_set_: [dtc, rfc, logr, sgdc]
            - minimal set of 4 estimators, chosen on the basis of speed
        accuracy_set_: [rfc, logr, knn, mlpc, rsvc]
            - set of 5 estimators, chosen on the basis of accuracy
        
    proba_predict: Boolean, optional, (default = False)
        Whether to predict probability estimates for all levels of target feature at base layer.  
        If False, predict class labels only.     
        
    incl_orig_input: Boolean, optional, (default = False)
        Whether to add the original data to the stack layer input.
        If False, predict on labels/probability estimates from the base estimators only.
    
    bootstrapping: Boolean, optional, (default = True)
        Whether to perform bootstrapping on the training data for the base estimators.  Applies to both producing 
        the training set for the stack estimator via k-fold cross validation, and the final fitting of base estimators.
        If False, no bootstrapping is performed at either point and the full original data is used for training
        
    kfolds: int, optional, (default = 5)
        Number of folds to be used in k-fold training of base estimator layer to fit stack layer

    random_seed: int, optional (default = None)
        Set as int for replicable results; leave as default 'None' for random.
    
    n_jobs: int, optional, (default = 1)
        The number of cores to use whenever parallel processing is available.
        
    verbose: Boolean, optional, (default = False)
        Whether to print information about what processes are happening as the super learner methods are being processed.
        If False, no details are printed during processing.
        
    Attributes
    ----------
    classes_ : array of shape = [n_classes] 
        The class/target feature labels (single output problem).
        
    self.base_estimators_ : list of shape = [n_estimators], containing strings representing estimator objects.
        The estimators to be used at layer one (base classifiers).
        These are a sub-class of attributes - 'etails in Parameters' 
        
    stack_estimator_ : string representing estimator object
        The estimator to be used at layer two (stack classifier)
        
    scores_ : array of shape = [n_folds, n_estimators]
        Accuracy scores from predicting base estimators for training stack layer, per validation fold


    Notes
    -----
    Ben Gorman's 'A Kaggler's Guide to Model Stacking in Practice' 
    (http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/)
    helped clarify for me how this algorithm should be structured, as did Sebastian Flennerhag's 
    'Introduction to Python Ensembles' (https://www.dataquest.io/blog/introduction-to-ensembles/).
    I incorporated a number of structural and python code elements from both Flennerhag's page and Max Halford's
    stacking algorithm (https://github.com/MaxHalford/xam/blob/master/xam/ensemble/stacking.py) in writing this 
    scikit-learn extension.
    
    See also
    --------
    
    ----------
    .. [1]  van der Laan, M., Polley, E. & Hubbard, A. (2007). 
            Super Learner. Statistical Applications in Genetics 
            and Molecular Biology, 6(1) 
            doi:10.2202/1544-6115.1309
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> clf = SuperLearnerClassifier()
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)

    """
    
    # Constructor for the classifier object
    def __init__(self, base_estimator_set = 'default_set', stack_estimator = 'dtc', proba_predict = False, \
                 incl_orig_input = False, bootstrapping = True, kfolds = 5, n_jobs = 1, random_seed = None, verbose = False):

        self.base_estimator_set = base_estimator_set
        self.stack_estimator = stack_estimator
        self.proba_predict = proba_predict
        self.incl_orig_input = incl_orig_input
        self.bootstrapping = bootstrapping
        self.kfolds = kfolds
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.verbose = verbose
        

    # The fit function to train a classifier
    def fit(self, X, y):
        """Build a SuperLearner classifier from the training set (X, y).
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. 
        y : array-like, shape = [n_samples] 
            The target values (class labels) as integers or strings.
        Returns
        -------
        
        Attributes
        -------
        
        self : object
        """     
        
        # Create attributes for estimators, with fixed hyperparameters
        
        # Hyperparameters derived from running gridsearch tests on the Fashion MNIST dataset per estimator
        # Adapt as required per use case
        
        dtc = DecisionTreeClassifier(criterion="entropy", max_depth = 9, min_samples_split=200)
        logr = LogisticRegression(C=0.3, max_iter=1000)
        knn = KNeighborsClassifier(n_neighbors=7, n_jobs=self.n_jobs)
        mlpc = MLPClassifier(alpha = 0.000001, hidden_layer_sizes=400, max_iter=2000)
        rsvc = SVC(C = 100000, gamma = 0.01, kernel="rbf", probability=True)
        sgdc = SGDClassifier(loss = 'modified_huber', alpha = 0.01, max_iter = 200, n_jobs=self.n_jobs)
        rfc = RandomForestClassifier(n_estimators=450, min_samples_split=200, n_jobs=self.n_jobs)
        # Note - GradientBoostingClassifier is too slow to be of practical use as part of a super learner on a standard desktop machine, 
        # so it is not included in any pre-made set of base estimators
        gbc = GradientBoostingClassifier(learning_rate = 0.1, max_depth = 5, min_samples_leaf = 70, \
                                         min_samples_split = 50, n_estimators = 350)

        # list of labels for estimators
        clf_labels = ['Decision Tree classifier', 'Random Forest classifier', 'Logistic Regression', \
                      'k-Nearest Neighbours classifier', 'Multi-layer Perceptron classifier', \
                      'Support Vector (rbf-kernel) classifier', 'SGD (modified huber) classifier', 'Gradient Boosting classifier']
        
        # list of all estimator attributes, corresponding to clf_labels
        clfs = [dtc, rfc, logr, knn, mlpc, rsvc, sgdc, gbc]
        
        # defining sets of base estimators as dictionary attributes
        
        # set of all estimators named in clf_labels
        all_clfs_ = {k:v for k,v in zip(clf_labels,clfs)}
        
        # set of estimators consisting of 'Random Forest classifier', 'Logistic Regression', 'k-Nearest Neighbours classifier', 
        # 'Multi-layer Perceptron classifier', 'C-Support Vector classifier', 'SGD classifier with modified huber loss function'
        default_set_ = {k:v for k,v in zip(clf_labels,clfs) if v not in {dtc, gbc}}
        
        # set of estimators consisting of 'Decision Tree classifier', 'Random Forest classifier', 'Logistic Regression',
        # 'SGD classifier with modified huber loss function'      
        speedy_set_ = {k:v for k,v in zip(clf_labels,clfs) if v not in {knn, mlpc, rsvc, gbc}}
        
        # set of estimators consisting of 'Random Forest classifier', 'Logistic Regression', 
        # 'k-Nearest Neighbours classifier', 'Multi-layer Perceptron classifier', 'C-Support Vector classifier'
        accuracy_set_ = {k:v for k,v in zip(clf_labels,clfs) if v not in {dtc, sgdc, gbc}}
        
        # Create user_set_ dictionary below to implement your own set of base estimators
        # Dictionary can include estimators as pre-defined above or as user-defined
        # First import any scikit-learn estimator modules that are not imported by default, e.g. AdaBoostClassifier()
        # Example:
        #     user_set_ = {'Random Forest classifier': rfc,
        #                  'Logistic Regression': LogisticRegression(C=0.5, max_iter=500),
        #                  'AdaBoost classifier': AdaBoostClassifier(),
        #                  'Random Forest classifier 2': RandomForestClassifier(n_estimators=1000, n_jobs=self.n_jobs)}
        user_set_ = default_set_
        
        # test_set = {k:v for k,v in zip(clf_labels,clfs) if v not in {rfc, knn, mlpc, rsvc}}
        
        # Convert the string base_estimator_set into an attribute reference
        
        if self.base_estimator_set == 'default_set':
            self.base_estimators_ = default_set_
        elif self.base_estimator_set == 'all_clfs':
            self.base_estimators_ = all_clfs_
        elif self.base_estimator_set == 'speedy_set':
            self.base_estimators_ = speedy_set_
        elif self.base_estimator_set == 'accuracy_set':
            self.base_estimators_ = accuracy_set_
        elif self.base_estimator_set == 'user_set':
            self.base_estimators_ = user_set_
        else:
            self.base_estimators_ = default_set_
        
        # Convert the string stack_estimator into an attribute reference
        
        if self.stack_estimator == 'dtc':
            self.stack_estimator_ = dtc
        elif self.stack_estimator == 'rfc':
            self.stack_estimator_ = rfc
        elif self.stack_estimator == 'logr':
            self.stack_estimator_ = logr
        elif self.stack_estimator == 'knn':
            self.stack_estimator_ = knn
        elif self.stack_estimator == 'mlpc':
            self.stack_estimator_ = mlpc
        elif self.stack_estimator == 'rsvc':
            self.stack_estimator_ = rsvc
        elif self.stack_estimator == 'sgdc':
            self.stack_estimator_ = sgdc
        else:
            self.stack_estimator_ = dtc
        
        # Check that X and y have the correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fitting
        self.classes_ = unique_labels(y)

        # Create empty arrays to store generated data to train stack layer and evaluate base learners
        
        # Prepare matrix to store predictions on all inputs, per validation fold, for training stack layer
        if self.proba_predict:
            kf_X = np.array([]).reshape(0, (len(self.base_estimators_)* len(self.classes_)))       
        else:
            kf_X = np.array([]).reshape(0, len(self.base_estimators_))
        # Prepare empty array to store target levels on all inputs, per validation fold, for training stack layer
        kf_y = np.array([], int)
        # Prepare matrix to store accuracy scores per base estimator per validation fold
        self.scores_ = np.zeros((self.kfolds, len(self.base_estimators_)))
        # Prepare empty list to store base estimator names in the order they're processed
        score_clfs = []
        
        if self.verbose:
            print("Generating layer one predictions for stack layer training \n")
            
        # Loop for generating kfold predictions

        for fold, (train_index, test_index) in enumerate(StratifiedKFold(n_splits=self.kfolds, shuffle=True).split(X, y)):
            
            # Check as to whether the the data inputted is in pandas dataframe/series or numpy ndarray form
            # and 
            if isinstance(X, pd.DataFrame):
                fold_xtrain, fold_xtest = X.iloc[train_index], X.iloc[test_index]
            else:
                fold_xtrain, fold_xtest = X[train_index], X[test_index]
            
            if isinstance(y, pd.Series):
                fold_ytrain, fold_ytest = y.iloc[train_index], y.iloc[test_index]
            else:
                fold_ytrain, fold_ytest = y[train_index], y[test_index]
                
            # Prepare matrix to store per-validation-fold predictions
            if self.proba_predict:
                fold_preds = np.zeros((fold_xtest.shape[0], len(self.base_estimators_)* len(self.classes_)))       
            else:
                fold_preds = np.zeros((fold_xtest.shape[0], len(self.base_estimators_)))
                
            # Loop for generating predictions per model
            
            for i, (nm, model) in enumerate(self.base_estimators_.items()):
                if self.verbose:
                    print('\t Processing', nm, 'on fold', (int(fold) + 1) )
                
                current_model = clone(model)
                
                # store base estimator names in the order they're processed
                if fold == 0:
                    score_clfs.append(nm)
                
                # Bootstrapping training data on fold, if required
                if self.bootstrapping:
                    fold_xtrain, fold_ytrain = resample(fold_xtrain, fold_ytrain, replace=True)
                    
                # fit current model on current training data    
                current_model.fit(fold_xtrain, fold_ytrain)
                
                # make predictions using current model
                if self.proba_predict:
                    current_model_fold_preds = current_model.predict_proba(fold_xtest)
                    # add predictions to array for this validation fold
                    for j, k in enumerate(range(len(self.classes_) * i, len(self.classes_) * (i + 1))):
                        fold_preds[:, k] = current_model_fold_preds[:, j]             
                else:
                    current_model_fold_preds = current_model.predict(fold_xtest)
                    # add predictions to array for this validation fold
                    fold_preds[:, i] = current_model_fold_preds
    #                print(fold_preds)
                
                # calculate accuracy of current model on this validation fold
                if self.proba_predict:
                    lb = LabelBinarizer().fit(y)
                    current_model_fold_score = accuracy_score(fold_ytest, lb.inverse_transform(current_model_fold_preds))
                else:
                    current_model_fold_score = accuracy_score(fold_ytest, current_model_fold_preds)
#                print(current_model_fold_score)
                
                # add accuracy scores to scores matrix
                self.scores_[fold, i] = current_model_fold_score
            
            # add fold predictions to stack-layer-training predictions array
            kf_X = np.r_[kf_X, fold_preds]
            # add fold target levels to stack-layer-training targets array
            kf_y = np.r_[kf_y, fold_ytest]
            if self.verbose:
                print("\t Fold", (int(fold) + 1), "processed.")
                print("\t Stack-layer training predictions matrix size after processing fold", (int(fold) + 1), "=", kf_X.shape, " \n")
                
        if self.verbose:
            print("Layer one training predictions completed. \n")
        
        # store scores_ as pandas dataframe
        self.scores_ = pd.DataFrame(self.scores_)
        self.scores_.columns = score_clfs
        if self.verbose:
            print("Accuracy score per estimator per fold:")
            print(self.scores_)      
        
        # add original input features to stack-layer-training predictions array if required
        if self.incl_orig_input:
            kf_X = np.c_[kf_X, X]

        # Train stack layer estimator
        if self.verbose:
            print("Fitting stack layer estimator...")
        self.stack_estimator_ = clone(self.stack_estimator_)
        self.stack_estimator_.fit(kf_X, kf_y)
        if self.verbose:
            print("Done \n")

        # Train final base learners
        if self.verbose:
            print("Fitting layer one estimators... \n")
            
        for i, (nm, model) in enumerate(self.base_estimators_.items()):
            if self.verbose:
                print('\t Processing', nm)
            # Bagging training data on fold, if required
            if self.bootstrapping:
                X, y = resample(X, y, replace=True)
            # fit current model on current training data    
            model.fit(X, y)

        if self.verbose:
            print("\n Super Learner fitting completed \n")
        
        # Return the classifier
        return self

    
    # The predict function to make a set of predictions for a set of query instances
    def predict(self, X):
        """Predict class labels of the input samples X.
        Parameters
        ----------
        X : array-like matrix of shape = [n_samples, n_features]
            The input samples. 
        Returns
        -------
        slclf_pred : array of shape = [n_samples, ].
            The predicted class labels of the input samples. 
        """
        
        # Check is fit had been called by confirming that the templates_ dictionary has been set up
        check_is_fitted(self, ['stack_estimator_', 'base_estimators_'])

        # Check that the input features match the type and shape of the training features
        X = check_array(X)

        # Prepare matrix to store first layer predictions for training second layer
        if self.proba_predict:
            pred_X = np.zeros((X.shape[0], (len(self.base_estimators_)* len(self.classes_))))      
        else:
            pred_X = np.zeros((X.shape[0], len(self.base_estimators_)))

        if self.verbose:
            print("Generating base learner predictions... \n")
        for i, (nm, model) in enumerate(self.base_estimators_.items()):
            # Bagging training data per estimator, if required
            if self.verbose:
                print('\t Processing base learner: ', nm)
            if self.proba_predict:
                model_pred_X = model.predict_proba(X)
                for j, k in enumerate(range(len(self.classes_) * i, len(self.classes_) * (i + 1))):
                    pred_X[:, k] = model_pred_X[:, j]               
            else:    
                model_pred_X = model.predict(X)
                pred_X[:, i] = model_pred_X
            
        # add original input features to stack-layer-training predictions array if required
        if self.incl_orig_input:
            pred_X = np.c_[pred_X, X]
            
        if self.verbose:
            print("\n Generating stack learner predictions... \n")
        
        slclf_pred = self.stack_estimator_.predict(pred_X)
        
        return slclf_pred
    
    def evaluate_base(self):
        
        """
        Provides information on base estimator accuracy scores.
        
        """
    
        check_is_fitted(self, ['scores_'])
    
        print("Accuracy score per estimator per fold:")
        print(self.scores_)
        
        print("Average accuracy score per estimator:")
        scores_avg = self.scores_.mean()
        print(scores_avg)

        ax = heatmap(self.scores_.corr(), annot=True)
        plt.show()
