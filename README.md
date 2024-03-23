# LIRecommend
The ultimate goal of this project is to design a scheme through which LinkedIn could improve it's job recommendation system.

Throughout this repository I will use the acronym POC to stand for 'Proof of Concept'. 

Currently, the work is being carried out as a capstone project for the UC San Diego Extended Studies Machine Learning Engineering Bootcamp, which I was enrolled in on November 6, 2023.   

The following were some initial goals or visions behind this project:

- can I come up with a way that could recommend jobs from LinkedIn that are more closely aligned with my preferences? Currently LinkedIn recommends jobs based on your user profile or a combination of it and keyword searching (through LinkedIn Jobs' search feature), but there is some degree of lack of accuracy or specificity to its engine.

- the initial vision for this project involved an AI that can recommend highly personalized, realistic, and practical career growth or learning paths to users by learning from their feedback on job postings. This idea was born from my own struggle to effectively prioritize and map out an action plan that would enable me to be a truly qualified candidate for the jobs I really want. Being still on the young and inexperienced side as a software engineer, it can feel very helpless and whimsical applying to dozens of appealing software engineering jobs which unfortunately are out of reach because I just don't have enough experience or the desired qualifications for the job. Applying to these jobs in mass helps me little. You may advise me to stick to applying to jobs that are within my wheelhouse so-to-speak, but the jobs that are within my wheelhouse are not the jobs I want. So instead of spending more and more time applying to jobs that I actually want, I think the time would be better spent beefing up the resume and gaining valuable experience. Applying to jobs is like fishing, and of course fishing is a numbers game, but you can't catch a nice Mahi Mahi with crappy bait. I feel confident enough in generalizing this to other people, particurly younger individuals or those just entering a given trade, to say the following: it can be extremely difficult to effectively map and navigate a career growth path because there are so many information sources, sub-optimal actions and avenues available to people that they hinder prioritization and general decision-making. The vast sea of career opportunities and paths, especially when viewed through a job postings site, is hard for one person to sail and finding the optimal approach can be an extremely tedious and draining process. Would AI be able to help lift some of this burden? Can we build an AI that could recommend a highly efficient and detailed series of steps to someone to increase their hirability for a set of jobs? It would involve first modeling or learning the user's job preferences or feedback to job postings, and then using that knowledge to either A) enhance or filter the job recommendations provided through a given site like LinkedIn or B) combine it with other data (like current skills or info extracted from a resume) from the user to make recommendations in a ChatGPT-like way, or both.

- for the project within the scope of the bootcamp, the natural and healthy goal is to just try to build something that perfoms A). 

- initially I was thinking of a tool or product that could work with postings from different websites/platforms, but for the purpose of data collection and standardization, I went on to restrict my focus to scraping exlusively LinkedIn Jobs postings.

- So the high level overview of what I am doing for the bootcamp assignment is the following:
- -  1. Manually go through job postings on LinkedIn, rate the job 0-3, 0 being completely outside of desire or radar, for me this would be a bus driver position for example, and 3 being extremely desirable, like an ML Engineering job that pays 160K a year. (Obviously there will be considerable biases here as my mentor pointed out, but they might be interesting to explore. And the goal is to in fact provide a POC for a methodology behind LinkedIn being able to model it's user's tendencies with more specificity, perhaps/hopefully suggesting better performance than LinkedIn currently, though defining performance is important and such a claim would require extensive testing (hundreds of users) to substantiate.) We will make a spreadsheet for the url for the job posting in one column and the rating in the next.
- -  2. After the spreadsheet with urls and ratings is made for a satisfactory number of postings, we make a script iterating over these rows in the spreadsheet that scrapes some of the page metadata using the url as well as the body text of the job posting and concatenates it all into a text file that is saved appropriately (all the text files should be saved to some designated file, and there should be some way to link it with the row number of the spreadsheet with the ratings, with a name or some other mechanism).
- -  3. Then we iterate over these text files, reading them, and plugging their text into a pre-engineered prompt that will ask ChatGPT to output a summary of different info from the text as a JSON dictionary, with specified fields such as salary, responsibilities, location, industries, etc. We save off our JSON files smartly and appropriately as we did their corresponding text files.
- -  4. Now we create a dataframe and iterate over the JSON files to keep appending data into the dataframe, one row for each JSON file. The dataframe will have a combination of numbers, lists, and NaN within it. The nature of the data in the different columns varies.
   
Now all of the above has been accomplished. Our goal right now is to train a model, but not before the data is further processed and we have defined our full feature engineering pipeline.    

One of the most pivotal decisions I had to make for this project is: how am I going to encode or represent the phrase data in such a way that I can train an accurate machine learning model on it? Many thoughts and ideas about this popped into my head that I had to ponder. They included:
   -- using a 1-1 function to map bags of words to unique numbers using Cantor's rule, which basically means the set of words that make up a phrase but not the ordering become important, and we will have many numbers to represent all the unique sets of words that make up our phrase space. This also means we would want to devise a scheme to treat these numbers as categorical variables rather than ordinal numbers, which probably means we would have to either one-hot or label encode the presence of a given number.
   -- this brings us to our next question: will the relevant columns share the same phrase space or will they have different phrase spaces thus requiring even more columns?
      --- my instinct is to avoid creating feature columns for each original column we are getting the phrases from. Why? There are two main reasons:
            A) we don't want too many columns in our dataframe as it requires more computational time and energy
            B) we assume that information that would normally be in one column could actually be stored in another, but the information is the same at the end of the day. More important than if the summary of a given category contains a given phrase is if the summary of all categories contain a given phrase as that means the original job posting contained that phrase. We are simply using ChatGPT to identify phrases from the original text that are notable and/or descriptive enough and put them into our predefined categories, but whether ChatGPT put a given piece of information under 'responsibilities' or 'job_function' shouldn't change how the model learns, or at least we won't worry about that for now in order to simplify the problem we are trying to solve. 
      --- my mentor recommended starting as simply as possible with TFIDF vectorization which I've considered doing, but in the end to get SOMETHING working I ended up doing the following:
         ---- I used a pretrained sentence transformer model called gte-base (see doc here: https://huggingface.co/thenlper/gte-base) to get embeddings for all my phrases , gte-base, which is a BERT-based model. For different sets of columns arranged based on shared corpus, meaning most of the columns like responsibilities and goals/objectives shared a corpus called 'commom' but some others like city had their own corpus, we calculated an embeddings map of all the phrases found in that set of columns, then performed KMeans clustering for a chosen number of clusters, giving cluster labels for all the phrases for that set of columns. Then, the number of each cluster label in each row is counted and that number is recorded in a column that gets added for each cluster. That is the trick behind most of the feature engineering I ended up carrying out. For the categorical label columns that didn't involve using a sentence transformer to bin items in the phrase space, like state, I simply made one-hot encodings of all the different categorical values found in the column. With my choices of cluster numbers for the corpi I binned, the data I initally passed to my models, which were a Decision Tree, Random Forest, and XGBoost had over 700 columns! This seems like too much and improved feature engineering would help solve this issue, but for now we will accept how our data is and try other algorithms. 

Not only could having 700 columns be detrimental to model performance, but it also makes the feature engineering step take way too long- in my case 16 minutes running it on a Jupyter notebook with a CPU, for only a 107 rows dataframe.

Anyway, I have evaluated various models on my small dataset (with the 700 or so columns) using two simple metrics: accuracy and mean absolute error. They are the following from best performing to worst performing: 
linear regression, accuracy: .68, MAE: .45
random forest classifier, accuracy: .59, MAE: .55
decision tree, accuracy: .59, MAE: .63
KNN Classifier, accuracy: .55, MAE: .5
KNN Regressor, accuracy: .55, MAE: .52
Support Vector Machine, accuracy: .55
random forest regression, accuracy: .5, MAE: .53
XGBoost, accuracy: .5, MAE: .59
Naive Bayes, accuracy: .45, MAE: .64

Then I tried dropping certain columns from my dataframe. The first experiment was to drop columns with extreme class inbalance. Basically columns where all but one of the datapoints fall in the majority class. I tried running the same algorithms on the dataframe with fewer columns to compare results to before. it turned out that the linear regression ended up doing much worse, and the Random Forest did the same.
removing columns with extreme class imbalance:
linear regression, accuracy: .45, MAE: .74
random forest, accuracy: .59, MAE: .55


         


