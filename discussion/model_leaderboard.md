| Model | FE Approach | Special hyperparams | accuracy | MAE |
|-------|------------|---------------------|----------|----------|
|  HGBC |   2     |   N/A |   72.7%  |   .318 |
| LinReg |   3    |   N/A |  63.6%  |   .480 |
|    RFC   |   Any    |   {'max_depth': None, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}  |   59.1%  |   .454  |
|   XGB    |   1     |   {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 200, 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 1.0} |   59.1%  |   .5  |
|    KNNR   |   1     |   n_neighbors=5  |   59.1%  |   .518  |
|    LinReg   |   2     |   N/A |   54.5%  |   .481  |
|    RFR   |    2     |  (max_depth=None, max_features='auto', min_samples_leaf=1, min_samples_split=2, n_estimators=300)   |   50%  |   .502  |
|  MCP  |   Any  |   N/A  |   46.7%  |   .654   |
| GNB |  1  |   N/A   |   36.4%   |    .773   |
|  RG   |   Any  |   N/A  |   25%    |   1.25   |

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

GNB - GaussianNaiveBayes

MCP - Majority Class Picker

RG - Random Guesser
