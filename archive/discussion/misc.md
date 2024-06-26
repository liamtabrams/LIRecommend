One of the most pivotal decisions I had to make for this project is: how am I going to encode or represent the phrase data in such a way that I can train an accurate machine learning model on it? Many thoughts and ideas about this popped into my head that I had to ponder. They included:
   -- using a 1-1 function to map bags of words to unique numbers using Cantor's rule, which basically means the set of words that make up a phrase but not the ordering become important, and we will have many numbers to represent all the unique sets of words that make up our phrase space. This also means we would want to devise a scheme to treat these numbers as categorical variables rather than ordinal numbers, which probably means we would have to either one-hot or label encode the presence of a given number.
   -- this brings us to our next question: will the relevant columns share the same phrase space or will they have different phrase spaces thus requiring even more columns?
      --- my instinct is to avoid creating feature columns for each original column we are getting the phrases from. Why? There are two main reasons:
            A) we don't want too many columns in our dataframe as it requires more computational time and energy
            B) we assume that information that would normally be in one column could actually be stored in another, but the information is the same at the end of the day. More important than if the summary of a given category contains a given phrase is if the summary of all categories contain a given phrase as that means the original job posting contained that phrase. We are simply using ChatGPT to identify phrases from the original text that are notable and/or descriptive enough and put them into our predefined categories, but whether ChatGPT put a given piece of information under 'responsibilities' or 'job_function' shouldn't change how the model learns, or at least we won't worry about that for now in order to simplify the problem we are trying to solve. 
      --- my mentor recommended starting as simply as possible with TFIDF vectorization which I experimented with as well, but I started off trying the following:
         ---- I used a pretrained sentence transformer model called gte-base, a BERT-based model (see doc here: https://huggingface.co/thenlper/gte-base) to get embeddings for all my phrases that were in the training set (at first I included test set data in the maps to compute clusters but that was not valid). For different sets of columns arranged based on shared corpus, meaning most of the columns like responsibilities and goals/objectives shared a corpus called 'common' but some others like city had their own corpus, we calculated an embeddings map of all the phrases found in that set of columns, then performed KMeans clustering for a chosen number of clusters, giving cluster labels for all the phrases for that set of columns. Then, the number of each cluster label in each row is counted and that number is recorded in a column that gets added for each cluster. That is the trick behind most of the feature engineering I ended up carrying out. For the categorical label columns that didn't involve using a sentence transformer to bin items in the phrase space, like 'state', I simply made one-hot encodings of all the different categorical values found in the column. With my choices of cluster numbers for the corpi I binned, the data I initally passed to my models, which were a Decision Tree, Random Forest, and XGBoost (and others, no spoilers!), had over 700 columns! This seems like too many and improved feature engineering would help solve this issue, but for now we will accept how our data is and try other algorithms. 

Not only could having 700 columns be detrimental to model performance, but it also makes the feature engineering step take way too long- in my case 16 minutes running it on a Jupyter notebook with a CPU, for only a 107 row dataframe.

Anyway, I evaluated various models on my small dataset (with the 700 or so columns) using two simple metrics: accuracy and mean absolute error. PLEASE NOTE THESE RESULTS ARE FOR THE INVALID TRIALS THAT HAD THE TEST SET PHRASES INCLUDED IN THE MAPS USED FOR CLUSTERING. THUS THEY ARE TECHNICALLY INVALID RESULTS BUT STILL INTERESTING TO LOOK AT. They were the following from best performing to worst performing: 
linear regression, accuracy: .68, MAE: .45

random forest classifier, accuracy: .59, MAE: .55

decision tree, accuracy: .59, MAE: .63

KNN Classifier, accuracy: .55, MAE: .5

KNN Regressor, accuracy: .55, MAE: .52

Support Vector Machine, accuracy: .55

random forest regression, accuracy: .5, MAE: .53

XGBoost, accuracy: .5, MAE: .59

Naive Bayes, accuracy: .45, MAE: .64

Then I tried dropping certain columns from my dataframe. The first experiment was to drop columns with extreme class inbalance. Basically columns where all but one of the datapoints fall in the majority class. I tried running the same algorithms on the dataframe with fewer columns to compare results to before. It turned out that the linear regression ended up doing much worse, and the Random Forest classifier did the same.
removing columns with extreme class imbalance:
linear regression, accuracy: .45, MAE: .74
random forest, accuracy: .59, MAE: .55

Then, I tried changing the number of common clusters defined in the feature engineering step. 300 and 650 clusters both yielded substantially worse results for the linear regression predictions than 500. However, 400 yielded an accuracy of nearly .64 with an MAE just over .42. So with 400 clusters, we got a better MAE but worse accuracy using rounded predictions from a linear regression model than we did using 500 clusters. Again all the findings up to this point used the inherently invalid approach of applying feature engineering approach 1 but with test set phrases influencing the clustering. 

Interestingly, the MAE from the Random Forest model also went down to just over .45, though the accuracy stayed at .59. We also perform a grid search on Random Forest Regressor, to see if there is anything gained from using that algorithm with 400 common cluster columns in our dataset. Once we feel exhausted enough with evaluating different minor tweaks to our feature engineering step, we can try to redesign our feature engineering step around word count and/or more likely TFIDF vectorization on our text files. If this ends up being the preferred way to feature engineer our data, then we will adopt this method. Then, we will move backward one more step to the original dataset, looking at the job postings themselves and the ratings. We will do some exploratory EDA, evaluate biases in our data, and possibly attempt a reproduction of the original ratings myself, to see if there is any drift in my own preferences or behavior. I may consider collecting more data to create a larger dataframe if time and energy permits.

These are my earliest findings from doing model evaluation and comparison combined with different feature engineering tweaks. I will discuss the more recent legitimate results from applying FE Approach 1 correctly and FE Approach 2 which I will clearly spell out in another file I'll call 'model evaluation'. 
