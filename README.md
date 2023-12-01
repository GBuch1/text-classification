# Snail

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/wXpymofm)

# Assignment 4: Text Classification
## Westmont College Fall 2023

# CS 128 Information Retrieval and Big Data

#### Assistant Professor Mike Ryu (mryu@westmont.edu)

## Author Information
#### Name(s): Darian Choi, Garret Buchanan 
#### Email(s): dchoi@westmont.edu, gbuchanan@westmont.edu

### **CORPUS AND WHY WE CHOSE IT**
#### We chose the corpus "Twitter samples" in order to see how the containment and length of words within a tweet impact the negativity, or positivity of a tweet. Also being able to see which words are the most impactful in giving a tweet a certain connotation.

### **PROBLEM DESCRIPTION**
#### Given a "Twitter samples" corpus, need to find a way to select and build impactful features that result in determining whether a given tweet has a positive or negative class, at a high accuracy rate. A prerequisite to this process is being able to create a classifier instance that can be trained with built featuresets and be able to predict classes based on the algorithm. a present informative features method must also be implemented in order to review the features that were most influential in determining the predicted class of a tweet. In order for all of this to be used, a runner file needs to be implemented in order to output the classifier accuracy as well as call the most informative features method. In order to ensure that the code is correct, sufficient unittests must also be written.

### **IMPLEMENTATION GUIDE**
To design a classifier we made three classes that were children classes from the starter code, OurFeature, OurFeatureSet, and OurClassifier to hold and process the data
1. OurFeature: we left Mikes's code as it is because it was sufficient and we did not preprocess the features here
2. OurFeatureSet: In implementing this we chose to just use a method build that builds a feature set.
To extract the features we chose from the raw string passed in from Twitter Samples, we tokenized each word and then extracted the length of the word as a feature and the word itself we also threw in :) and :( as features which will be explained further down below.
Some key design decisions that were made in this class were that we initially tried to just use the split method but realized that it did not get all the words by itself so we tried to use Regex but regex we ran into some testing errors so we finally found that using the .isalpha and checking if the word length is more then 1 was the most simple way of implementing the tokenizer.
4. OurClassifier: (gamma, Present features, train)\
**gamma**: to implement gamma, we took in a FeatureSet and then created 2 holder variables to hold the max scores and predicted class. Then iterating through each feature's probability for the specific class using log addition and repeating then choosing which class had the highest probability and returning that class. One key error I ran into was the log addition where I needed to add 1 to the probability before adding.\
**present features**: To present features, we started  by finding ratios between the positive and negative probabilities to determine which features were the most informative. then we added them to a list and sorted the tuples by the ratios using a lambda function. The biggest key design decision we had to make here was getting around the division by 0 error which we had to just not count features if one of the classes had 0 as a probability.\
**train**: finally our train loops through all the feature sets in the training set passed in. We made a counter for the total amount of features in the class and the frequency of the feature in the class. Then after getting the numbers for each feature we iterate through each feature and divide the two counter values to get a probability of a feature belonging to a specific given class.\

TESTS IMPLEMENTATION: 
FeatureSetTest: we set up a lot of features and then built our own feature set and through calling the build method on the same features we compared to see if the values were equivalent and being built correctly.
GammaTests: gamma was tested by feeding a feature set that we knew would be a certain class based on the math we did and then comparing if that class matched the predicted class gamma outputted 
present_featurestest: Because present features returned nothing we needed to create a different method called return_present_features (contained all the same logic from the original present features but the only difference being that it returns a string) in order to test our method. In doing this we checked if the return value was equivalent to the expected string based on the classifier instance we had passed in. 


### **MOST INFORMATIVE FEATURES ANALYSIS**
#### The features we chose to implement within our build are the length of a word within a given tweet, if a word is contained within a given tweet, and if a tweet contains a ":)" (smiley face), or ":(" (frowny face). We chose the "length" feature because we assumed that shorter words have more negative connotations (bad, mad, sad) and that longer words have more positive connotations (beautiful, exquisite, sensational). We chose the "contains word" feature because it is very telling if a tweet is positive or negative based on if it contains word(s) that have a positive or negative sentiment attached to them the majority of the time. For example, the word "sad" is going to have a negative connotation associated with it much more often than a positive association on average. The containment of the ":)" and ":(" feature, which is implemented within the "contains" feature code, is added because the use of a smiley and frowny face is extremely informative in determining whether a tweet is positive or negative. We can explain the classifier accuracy because tweets with a “:)” are rarely ever a negative tweet and vice versa. This feature along with the "length" and "contains word" features, result in a classifier with about 88% percent accuracy for the "Twitter samples" corpus. We can see that the accuracy is so high when viewing the top 30 most informative features within our output. 

### **PARTNER GROUP WORK DIVISION**
#### Throughout this entire process both of us were present when implementing the entire project. Darian did most of the models and runner portion of the assignment and Garrett did most of the unittest writing. Throughout the duration of the assignment however we both helped each other on different aspects of the code we were working on.

### **WHAT WE LEARNED WORKING AS A GROUP**
#### What was beneficial when working as a group was that when either of us ran into a problem, there was someone there to bounce ideas off of. This was helpful because it provided the opportunity to think through problems without having to wait for outside help, which aided in finishing this assignment on time and in a correct manner. We also learned that working as a group can be difficult with managing github restrictions when it comes to pushing and pulling code, but through that struggle we now are much more well equipped going forward into future projects. Also an important aspect of working as a team was being able to have meaningful communication on a consistent basis, whether that was in person or electronically in order to communicate things such as when and where we would meet, or what we were changing within the code.

