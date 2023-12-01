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
#### Given a "Twitter samples" corpus, need to find a way to select and build inmpactful features that result in determining whether a given tweet has a positive or negative class, at a high accuracy rate. A prerequisite to this process is being able to create a classifier instance that can be trained with built featuresets and be able to predict classes based on the algorithim. a present informative features method must also be implemented in order to review the features that were most influential in determining the predicted class of a tweet. In order for all of this to be used, a runner file needs to be implemnted in order to output the classifier accuracy as well as call the most informative features method. In order to ensure that the code is correct, sufficient unittests must also be written.

### **IMPLEMENTATION GUIDE**
"TODO:"

### **MOST INFORMATIVE FEATURES ANALYSIS**
#### The features we chose to implement within our build are the length of a word within a given tweet, if a word is contained within a given tweet, and if a tweet contains a ":)" (smiley face), or ":(" (frowny face). We chose the "length" feature because we assumed that shorter words have more negative connotations (bad, mad, sad) and that longer words have a more positive connotation (beautiful, exquisite, sensational). We chose the "contains word" feature because it is very telling if a tweet is positive or negative based on if it contains word(s) that have a positive or negative sentiment attached to them the majority of the time. For example the word "sad" is going to have a negative connotation associated with it much more often than a positive association on average. The containment of the ":)" and ":(" feature, which is implemented within the "contains" feature code, is added because the use of a smiley and frowny face is extremely informative in determining whether a tweet is positive or negative. We can explain the classifier accuracy because tweets with a “:)” are rarely ever a negative tweet and vice versa. This feature along with the "length" and "contains word" features, result in a classifier with about 88% percent accuracy for the "Twitter samples" corpus. We can see that the accuracy is so high when viewing the top 30 most informative features within our output. 

### **PARTNER GROUP WORK DIVISION**
#### Throughout this entire process both of us were present when implementing the entire project. Darian did most of the models and runner portion of the assignment and Garrett did most of the unittest writing. Throughout the duration of the assignment however we both helped each other on different aspects of the code we were working on.

### **WHAT WE LEARNED WORKING AS A GROUP**
#### What was beneficial when working as a group was that when either of us ran into a problem, there was someone there to bounce ideas off of. This was helpful because it provided the opportunity to think through problems without having to wait for outside help, which aided in finishing this assignment on time and in a correct manner. We also learned that working as a group can be difficult with managing github restrictions when it comes to pushing and pulling code, but through that struggle we now are much more well equipped going forward into future projects. Also an important aspect of working as a team was being able to have meaningful communication on a consistent basis, whether that was in person or electronically in order to communicate things such as when and where we would meet, or what we were changing within the code.

