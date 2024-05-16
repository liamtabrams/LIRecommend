# LIRecommend

## Summary of important/notable files and folders

-**web_app_proto/ML_API** is where I am currently developing the FastAPI application I am deploying inside a Docker container

-**parse_li_jobs.ipynb** is the notebook I did most of the early prototyping a preliminary data analyis in. It is quite a massive notebook and will likely need to be broken up into multiple notebooks so that the reader isn't overwhelmed having to scroll through this one notebook. 

-**LIRecommend_ModelEvaluation.ipynb** is where I did model evaluation for a large handful of different architectures and witnessed how they compared to each other in terms of accuracy with three different feature engineering approaches. The results of those experiments are shown in tables in discussion/model_leaderboard.md   

-**pre-trained_models** is an, I feel, aptly-named folder for, well, pretrained models, ones that I decide to checkin because they are options for pre-loaded models in the deployed app or due to some other reason. I will try to avoid checking things in just because I feel like it and I will also make an effort to trim the fat of this repo by removing unnecessary or unused code wherever warranted, and to in general clean up the presentation and organization of this repo. 

## The Vision and Overview

The ultimate goal of this project is to design a scheme through which LinkedIn could improve it's job recommendation system.

Throughout this repository I will use the acronym POC to stand for 'Proof of Concept'. 

Currently, the work is being carried out as a capstone project for the UC San Diego Extended Studies Machine Learning Engineering Bootcamp.   

The following were some initial goals or visions behind this project:

- can I come up with a way that could recommend jobs from LinkedIn that are more closely aligned with my preferences? Currently LinkedIn recommends jobs based on your user profile or a combination of it and keyword searching (through LinkedIn Jobs' search feature), but there is some degree of lack of accuracy or specificity to its engine. UPDATE: I have recently discovered a new feature in LinkedIn Jobs where it asks you a question like 'How many years experience do you have with Scala?' which it can use as a very good filter in its recommendation engine which is geared toward matching jobs with an applicant's current skillset or wheelhouse so to speak. This objective is not necessarily baked into the algorithm that recommends jobs in the tool I built. Instead, the algorithm models how much a user desires a particular job. More research into the following claim is needed: the jobs a user is most qualified for or has the highest likelihood of currently getting hired into are typically not the jobs they would rate most highly in terms of desirability. For someone on the hunt and needing rapid re-employment, the strategy behind LinkedIn's recommendation engine is much more appropriate. But for someone looking for assistance mapping out their desired career progression or finding the optimal path to take to become qualified for jobs they want the most, being able to predict job desirability, which in many cases barring interest or qualifications is largely affected by something like salary, is better in a system where the user needs help understanding a good approach they can take to make their skills or the answers to questions that LinkedIn provides match up best with the jobs they rated as most desirable. So it does still seem that LinkedIn could benefit from having an additional mechanism for users to rate or give feedback to individual job postings. A potentially sophisticated and state-of-the-art career AI assistant system powered via LinkedIn or any similar platform could essentially utilize both the "qualifications" model with the "desirability" model to find extremely good path choices and be able to give recommendations with necessary amounts of detail that could help the user plan and prioritize the steps in their career progression.   

- the initial vision for this project involved an AI that can recommend highly personalized, realistic, and practical career growth or learning paths to users by learning from their feedback on job postings. This idea was born from my own struggle to effectively prioritize and map out an action plan that would enable me to be a truly qualified candidate for the jobs I really want. Being still on the young and inexperienced side as a software engineer, it can feel very helpless and whimsical applying to dozens of appealing software engineering jobs which unfortunately are out of reach because I just don't have enough experience or the desired qualifications for the job. Applying to these jobs in mass helps me little. You may advise me to stick to applying to jobs that are within my wheelhouse so-to-speak, but the jobs that are within my wheelhouse are not the jobs I want. So instead of spending more and more time applying to jobs that I actually want, I think the time would be better spent beefing up the resume and gaining valuable experience. Applying to jobs is like fishing, and of course fishing is a numbers game, but you can't catch a nice Mahi Mahi with crappy bait. I feel confident enough in generalizing this to other people, particularly younger individuals or those just entering a given trade, to say the following: it can be extremely difficult to effectively map and navigate a career growth path because there are so many information sources, sub-optimal actions and avenues available to people that they hinder prioritization and general decision-making. The vast sea of career opportunities and paths, especially when viewed through a job postings site, is hard for one person to sail and finding the optimal approach can be an extremely tedious and draining process. Would AI be able to help lift some of this burden? Can we build an AI that could recommend a highly efficient and detailed series of steps to someone to increase their hirability for a set of jobs? It would involve first modeling or learning the user's job preferences or feedback to job postings, and then using that knowledge to make predictions to either A) do something simple like enhance or filter the job recommendations provided through a given site like LinkedIn or go even further and B) build the kind of thing I'm sort of daydreaming on by combining these predictions with other data (like current skills or info extracted from a resume) from the user to make recommendations in a ChatGPT-like way.

- for the project within the scope of the bootcamp, the natural and healthy goal is to just try to build something that perfoms A). 

- initially I was thinking of a tool or product that could work with postings from different websites/platforms, but for the purpose of data collection and standardization, I went on to restrict my focus to scraping exclusively LinkedIn Jobs postings.

- So the high level overview of what I am doing for the bootcamp assignment is the following:
- -  1. Manually go through job postings on LinkedIn, rate the job 0-3, 0 being completely outside of desire or radar, for me this would be a bus driver position for example, and 3 being extremely desirable, like an ML Engineering job that pays 160K a year. (Obviously there will be considerable biases here as my mentor pointed out, but they might be interesting to explore. And the goal is to in fact provide a POC for a methodology behind LinkedIn being able to model it's user's tendencies with more specificity, perhaps/hopefully suggesting better performance than LinkedIn currently, though defining performance is important and such a claim would require extensive testing (hundreds of users) to substantiate.) We will make a spreadsheet for the url for the job posting in one column and the rating in the next.
- -  2. After the spreadsheet with urls and ratings is made for a satisfactory number of postings, we make a script iterating over these rows in the spreadsheet that scrapes some of the page metadata using the url as well as the body text of the job posting and concatenates it all into a text file that is saved appropriately (all the text files should be saved to some designated file, and there should be some way to link it with the row number of the spreadsheet with the ratings, with a name or some other mechanism).
- -  3. Then we iterate over these text files, reading them, and plugging their text into a pre-engineered prompt that will ask ChatGPT to output a summary of different info from the text as a JSON dictionary, with specified fields such as salary, responsibilities, location, industries, etc. We save off our JSON files smartly and appropriately as we did their corresponding text files.
- -  4. Now we create a dataframe and iterate over the JSON files to keep appending data into the dataframe, one row for each JSON file. The dataframe will have a combination of numbers, lists, and NaN within it. The nature of the data in the different columns varies.
   
Now all of the above has been accomplished. Our goal right now is to train a model, but not before the data is further processed and we have defined our full feature engineering pipeline. See 'misc.txt' in the discussion folder about my thought process and findings when it came to experimenting with different feature engineering approaches and models. UPDATE: although it was heavily considered and experimented with in the feature engineering step as placeholder LLM, as of right now the only role ChatGPT is playing on the server side is extracting the salary info from the job posting and converting it to the desired format before appending a row to the user's labelled dataset. This isn't trivial however the ChatGPT/LLM bit may be ditched altogether if I find the added performance from it is not worth the cost it accrues, which for me has been on the order of about 30 cents per 1000 API calls. It's not a huge cost, but there could probably be logic written that could do as good of a job as ChatGPT if not better at getting the salary data all nice and clean and in the right units. 

UPDATE: I am currently developing the web tool as a FastAPI application deployed in a Docker container. The web tool currently consists of three pages: a home page, a page for predicting, and a page for collecting and reviewing labelled data as well as the current model's accuracy. 
