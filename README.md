# SuperLearner

Scikit-learn extension implementing SuperLearner Classifier.

SLC.py contains the class SuperLearnerClassifier(), a functioning SuperLearner Classifier.

The Super Learner is a heterogeneous stacked ensemble classifier, a classification model that uses a set of base classifiers of different types, the outputs of which are then combined in another classifier at the stacked layer.  To avoid overfitting the generation of the stacked layer training set uses a k-fold cross validation process.

As a Scikit-learn extension, a number of scikit-learn base estimator implementations are utilised.  These are:

 * 'dtc' - Decision Tree classifier
 * 'rfc'- Random Forest classifier
 * 'logr'- Logistic Regression
 * 'knn'- k-Nearest Neighbours classifier
 * 'mlpc'- Multi-layer Perceptron classifier
 * 'rsvc'- C-Support Vector Classification with rbf kernel
 * 'sgdc'- Stochastic Gradient Descent classifier with modified huber loss function
 
User has the option of any of these at the stack level and a few prescribed cominations at the base level.  User also has the option to choose number of k-folds, whether to perform bootstrapping on the training data for the base estimators, and whether to add the original data to the stack layer input.

See SLC_demo.ipynb for example of SuperLearner applied to the [Fashion MNIST dataset](https://www.kaggle.com/zalando-research/fashionmnist).

SuperLearnerClassifier() parameters are:

    stack_estimator: string, optional, (default = 'dtc')
        The estimator used at the stack layer.  Options are as above
        
    base_estimator_set: string representing list, optional, (default = 'default_set')
        A list of base estimators used to train the stack layer.  Available options are:
        all_clfs_: [dtc, rfc, logr, knn, mlpc, rsvc, sgdc] 
            - All 7 available estimators.
        default_set_: [rfc, logr, knn, mlpc, rsvc, sgdc] 
            - set of 6 estimators with dtc omitted as it is used at stack layer by default.
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
    
    n_jobs: int, optional, (default = 1)
        The number of cores to use whenever parallel processing is available.
        
    verbose: Boolean, optional, (default = False)
        Whether to print information about what processes are happening as the super learner methods are being processed.  
        If False, no details are printed during processing.
        
        
SuperLearnerClassifier() methods are:

    fit(X, y), to train a model, with parameters:
            X : array-like, shape = [n_samples, n_features]
                The training input samples. 
            y : array-like, shape = [n_samples] 
                The target values (class labels) as integers or strings.            
        Returns object - the trained model

    predict(X), to predict class labels of the input samples X, with parameter:
            X : array-like matrix of shape = [n_instances, n_features]
                The input samples. 
        Returns array of shape = [n_instances, ]. The predicted output values for the input samples. 
        
    evaluate_base(), provides information on trained base estimator accuracy scores.
    

    
To do:
 * Check bootstrapping implementation
 * Implement regression
 * Implement more flexible method of choosing base estimator set
