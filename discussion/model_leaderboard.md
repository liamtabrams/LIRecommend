| Model | FE Approach | Special hyperparams | accuracy | MAE |
|-------|------------|---------------------|----------|----------|
|  HGBC |   2     |   N/A |   72.7%  |   .318 |
| LinReg |   3    |   N/A |  63.6%  |   .480 |
|       |   Row 3     |   Row 3  |   Row 3  |   Row 3  |
|       |   Row 4     |   Row 4  |   Row 4  |   Row 4  |
|       |   Row 5     |   Row 5  |   Row 5  |   Row 5  |
|  MCP  |   Any  |   N/A  |   46.7%  |   .654   |
|  RG   |   Any  |   N/A  |   25%    |   1.25   |

Glossary

FE approach 1 - use CHATGPT to summarize postings in the form of JSON files, then perform clustering on phrases in the training set and create cluster count features. Dataframe has salary features and cluster count features, as well as city and state features.

FE approach 2 - perform basic TFIDF vectorization on the 'posting_text' column after doing the train-test split. Then use just the TFIDF matrix as the features matrix for training and making predictions with. 

FE approach 3 - combine mostly FE approach 2 with early steps to generate the dataframe in FE approach 1, ultimately combining the TFIDF matrices with the min_ and max_salary columns from the dataframe that FE approach 1 gets applied to.  

HGBC - HistGradientBoostingClassifier

LinReg - Linear Regression

RFC - Random Forest Classifier

DT - Decision tree

SVC - Support Vector Machine

GNB - GaussianNaiveBayes

MCP - Majority Class Picker

RG - Random Guesser
