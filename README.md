# MACHINE LEARNING FINAL PROJECT

## Goal of This Project
* We want to predict an output of a [bank marketing](bank-additional-full.csv) dataset.

## The Dataset
* 41000+ Examples of data.
* 20 attributes: 10 numerical and 10 categorical.
* Parts within the dataset:
    * Personal information:
        * Age, Job, Education, etc...
    * Bank account information:
        * Balance, if has a loan.
    * Contact information
        * When client last time contact bank.
* Output:
    * Has the client subscribed a term deposit?

## Challenges Presented With This Project
* There are many categorical attributes.
  * We would need to convert these to numbers.
  * Normalization
* String data that contains meaningless punctuation parts e.g. ("").
* Some unknown values.

## The Type of Model Chosen for the Project
* Logistic Regression
  
## Possible Problems With the Model Chosen
* Over-fitting?
    * Not likely
    * 31000+ examples
    * Relatively small amount of features.

# Approaches Used

## First Approach 
* **Numpy and Pandas** is used 
    * In this approach we have to preprocess the data.
  
## Second Approach
* **Pandas and Sklearn** libraries are used.
*  In this approach we combine the variables in features to make one
   *  Also add a new column
<!-- TODO: NEED TO PLACE ACCURACY'S -->
# Accuracy's Reported

## First Approach
* Train accuracy: **91.10%**
* Test accuracy: **91.65%**
## Second Approach
* Train accuracy: **91.21%**
* Test accuracy: **91.35%**

# Conclusion to the Project
* About 3.4% of clients have subscribed a term deposit
* Both the second model and the third model have a good performance
* The importance of data preprocessing for dirty dataset
  * Increase significantly amount of accuracy
  * May increase the efficiency to train the model
