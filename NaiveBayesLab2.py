from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as pp

# Summary: Creates a bar graph based of the digit images and which digit they correspond to
# param1: <name>countedImages</name> expecting a dictionary / function that returns a dictionary.  Uses keys and
#                                    values to create bar graph of categories
def createBarGraph(countedImages):
    pp.bar(range(len(countedImages)), list(countedImages.values()), align= 'center')
    pp.xticks(range(len(countedImages)), list(countedImages.keys()))
    pp.xlabel('Category')
    pp.title('Categorical Data for Digits Dataset')
    pp.ylabel('Count of Images')

    pp.show()


# Summary: Counts the number of times a digit is represented in the dataset
# param1: <name>y</name> expecting an array of digits represented in the dataset
# returns: dictionary of the digits and the number of times they are present in the dataset
def countCategories(y):
    countOfImages = {}
    for digit in y:
        if digit not in countOfImages:
            countOfImages[digit] = 0
        countOfImages[digit] = countOfImages[digit] + 1

    return countOfImages

# Loads dataset
digits_dataset = datasets.load_digits()
x = digits_dataset.data
y = digits_dataset.target

# Split data into test and training set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# MultinomialNB
# used Multinomial over Gaussian because it has better / more consistent accuracy
mnb = MultinomialNB()
# train
mnb.fit(x_train, y_train)
# predict test set
predict_values = mnb.predict(x_test)

# find accuracy of model
print("Number of mislabeled points out of " + str(x_test.shape[0]) + " points : " + str((y_test != predict_values).sum()))
accuracy = accuracy_score(y_test, predict_values, normalize=True)
rounded = round(accuracy, 2)
print("accuracy: " + str(rounded))

createBarGraph(countCategories(y))