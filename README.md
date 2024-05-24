# Evaluating Performance of Semi-Supervised Self Training in Identifying Fake Reviews

### 0 TL;DR summary 
This project utilizes self-training (a semi-supervised learning method) combined with multiple supervised learning algorithms as base learners to identify "fake" Yelp reviews. Three different supervised learning methods are used as base learners – Naïve Bayes, Decision Trees and Logistic Regression.

### 1 Background and Introduction
The main objective of our project is to build classifiers using Semi-Supervised learning methods. We will then use this classifier to identify “fake” restaurant reviews posted on Yelp. Yelp is a website which publishes crowd-sourced reviews about local businesses including restaurants [7]. Yelp uses its own proprietary algorithm for filtering “fake” reviews. For the purpose of this project, we would be assuming Yelp classification as pseudo ground truth. Semi-supervised learning is a class of supervised learning tasks and techniques that also make use of unlabeled data for training - typically a small amount of with a large amount of unlabeled data. Supervised learning methods are effective when there are sufficient labeled instances to construct classifiers.
Labeled instances are often difficult, expensive, or time consuming to obtain, because they require empirical research. When it comes to restaurant reviews, we have a large supply of unlabeled data. Often semi supervised learning achieves a better accuracy than supervised learning which is only trained on the labeled data [6]. 

There are various approaches that can be used for semi-supervised learning. These include Expectation Maximization, Graph Based Mixture Models, Self-Training and Co-Training methods. In our project, we will be focusing on applying the Self-Training approach to Yelp’s reviews. In self-training, the learning process employs its own predictions to teach itself. An advantage of self-training is that it can be easily combined with any supervised learning algorithm as base learner [6]. 

We will be using three different supervised learning methods - Naïve Bayes, Decision Trees and Logistic Regression as base learners. We would then be comparing the accuracy of each of the semi-supervised learning methods with its respective base learner. The base learners would be using both behavioral and linguistic features.

#### 1.1 Related Work
Extensive studies (Mukherjee et al., 2013; Liu et al., 2013) have been done on determining the
effectiveness of existing research methods in detecting real-life fake reviews on a commercial
website like Yelp and in trying to emulate Yelp’s fake review filtering algorithm.

Apart from this, Liu et al., (2013) proposed a novel model to detect opinion spamming in a
Bayesian framework and model the spamicity of reviewers by identifying certain behavioral
features. The key motivation is based on the hypothesis that opinion spammers differ from others
on behavioral dimensions [2].

Research has also been done (Jafar et al., 2015) in the application of semi supervised learning to a
pool of unlabeled data and augmenting performance of supervised learning algorithm. They have
studied the semi-supervised self-training algorithm with decision trees as base learners.

### 2 Proposed Method
Extensive studies have already been done on detecting spam using supervised learning techniques.
Mukherjee et al., (2013) have built upon this by using Yelp’s classification of the reviews as
pseudo ground truth. Additionally, Li et al., (2011) have used semi supervised co-training on
manually labeled dataset of fake and non-fake reviews.

For our project, we will be focusing on applying semi-supervised self-training to Yelp’s reviews
by using Yelp’s classification as pseudo ground truth. Our approach is inspired from the above
two state of art research on review classification.

We aim to come up with a new solution that will help increase the performance of semi-supervised
approach – the idea being that semi-supervised learning methods could improve upon the
performance of supervised learning methods in the presence of unlabeled data.

To test this hypothesis, we implemented the self-training algorithm using Naïve Bayes, Decision
Trees and Logistic Regression as base learners and compared their performance.

#### 2.1 Data Collection
We built a Python crawler to collect restaurant reviews from Yelp. Reviews were collected for all
restaurants in a particular zip code in New York. We collected both the recommended and non-
recommended reviews as classified by Yelp. The dataset consists of approximately 40k unique
reviews, 30k users and 140 restaurants. The following attributes were extracted:

- Restaurant Name
- Average Rating
- User Name
- Review Text
- Rating
- Date of Review
- Classification by Yelp (Recommended / Not Recommended)

#### 2.2 Data Preprocessing
We carried out the following steps during preprocessing:

- Data cleaning

The data that we collected had lots of duplicate records and the first step was to remove
these. Following this, we modified the date field of all the records to ensure that the
formatting was consistent.

- Text processing

The first step here was to remove all the Stop Words. Stop Words are words which do not
contain important significance to be used in search queries. These words are filtered out
because they return vast amount of unnecessary information [8]. Then we converted the
text to lower case and removed punctuations, special characters, white spaces, numbers
and common word endings. Finally, we created the Term Document Matrix to find
similarity between the text reviews.




