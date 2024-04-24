**Cross validation will be needed to more confidently assess and rank each of these models. The following are results for one train-test split at random_seed=42. I will make another table for averages of cross validation below this**

| Model | FE Approach | Special hyperparams | accuracy | MAE |
|-------|------------|---------------------|----------|----------|
|  HGBC |   2 & 3   |   default ({'categorical_features': None, 'class_weight': None, 'early_stopping': 'auto', 'interaction_cst': None, 'l2_regularization': 0.0, 'learning_rate': 0.1, 'loss': 'log_loss', 'max_bins': 255, 'max_depth': None, 'max_iter': 100, 'max_leaf_nodes': 31, 'min_samples_leaf': 20, 'monotonic_cst': None, 'n_iter_no_change': 10, 'random_state': None, 'scoring': 'loss', 'tol': 1e-07, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False})|   72.7%  |   .318 |
| LinReg |   3    |   N/A |  63.6%  |   .480 |
| KNNR |   3    |   n_neighbors=5 |  63.6%  |   .482 |
| XGB |   3    |   default |  59.1%  |   .409 |
|    RFC   |   Any    |   {'max_depth': None, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}  |   59.1%  |   .454  |
|   XGB    |   1     |   {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 200, 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 1.0} |   59.1%  |   .5  |
|    KNNR   |   1     |   n_neighbors=5  |   59.1%  |   .518  |
|    LinReg   |   1     |   N/A  |   59.1%  |   .614  |
|    LinReg   |   2     |   N/A |   54.5%  |   .481  |
|    XGB   |   2     |   default |   54.5%  |   .5  |
|    DT   |   1     |   default |   54.5%  |   .5  |
|    KNNC   |   1     |   n_neighbors=5 |   54.5%  |   .5  |
|    SVC   |   Any    |   default |   54.5%  |   N/A  |
|    RFR   |    2     |  (max_depth=None, max_features='auto', min_samples_leaf=1, min_samples_split=2, n_estimators=300)   |   50%  |   .502  |
| GNB |  2  |   default   |   50%   |    .545   |
| KNNC |  3  |   n_neighbors=5   |   50%   |    .591   |
|    DT   |    2     |  default   |   50%  |   .636  |
|    HGBC   |    1     |  default   |   50%  |   .682  |
|  MCP  |   Any  |   N/A  |   46.7%  |   .654   |
| GNB |  1  |   default   |   36.4%   |    .773   |
|  DT  |   3  |   default  |   31.8%  |   .818   |
|  RG   |   Any  |   N/A  |   25%    |   1.25   |

**Results of 5-fold cross validation using same train-test splits after processing data with FE Approach 3. Included are random_seeds 42, 95, 13, 21, and 507. **

| Model | FE Approach | Special hyperparams | accuracy | MAE |
|-------|------------|---------------------|----------|----------|
| LinReg |   3    |   N/A |  66.4%  |   .474 |
| HGBC |   3    |   default |  58.2%  |   .455 |
| XGB |   3    |   default |  55.5%  |   .482 |
| RFC |   3    |   (max_depth=None, max_features='auto', min_samples_leaf=1, min_samples_split=2, n_estimators=300) |  53.6%  |   .527 |
| SVC |   3    |   default |  50.9%  |   N/A |
| KNNR |   3    |   n_neighbors=5 |  49.1%  |   .571 |
| KNNC |   3    |   n_neighbors=5 |  46.4%  |   .609 |
| DT |   3    |   default |  39.1%  |   .682 |

With feature engineering approach 3, Linear Regression did the best, and it's accuracy and error scores had by far the least variance when I did the cross validation.

**Results of 5-fold cross validation using same train-test splits after processing data with FE Approach 1. Included are random_seeds 42, 95, 13, 21, and 507. **

| Model | FE Approach | Special hyperparams | accuracy | MAE |
|-------|------------|---------------------|----------|----------|
| LinReg |   1    |   N/A |  26.4%  |   >1000000000 |
| HGBC |   1    |   default |  46.4%  |   .691 |
| XGB |   1    |   default |  57.3%  |   .545 |
| RFC |   3    |   (max_depth=None, max_features='auto', min_samples_leaf=1, min_samples_split=2, n_estimators=300) |  53.6%  |   .527 |
| SVC |   3    |   default |  50.9%  |   N/A |
| KNNR |   3    |   n_neighbors=5 |  49.1%  |   .571 |
| KNNC |   3    |   n_neighbors=5 |  46.4%  |   .609 |
| DT |   1    |   default |  45.5%  |   .700 |

With feature engineering approach 1, Linear Regression did the best, and it's accuracy and error scores had by far the least variance when I did the cross validation.


# Glossary

FE approach 1 - use CHATGPT to summarize postings in the form of JSON files, then perform clustering on phrases in the training set and create cluster count features. Dataframe has salary features and cluster count features, as well as city and state features.

FE approach 2 - perform basic TFIDF vectorization on the 'posting_text' column after doing the train-test split. Then use just the TFIDF matrix as the features matrix for training and making predictions with. 

FE approach 3 - combine mostly FE approach 2 with early steps to generate the dataframe in FE approach 1, ultimately combining the TFIDF matrices with the min_ and max_salary columns from the dataframe that FE approach 1 gets applied to.  

HGBC - HistGradientBoostingClassifier

LinReg - Linear Regression (note that predictions are rounded before calculating accuracy)

RFC - Random Forest Classifier

RFR - Random Forest Regressor

DT - Decision tree

SVC - Support Vector Machine

KNNR - K Nearest Neighbors Regressor

KNNC - K Nearest Neighbors Classifier

GNB - GaussianNaiveBayes

MCP - Majority Class Picker

RG - Random Guesser
