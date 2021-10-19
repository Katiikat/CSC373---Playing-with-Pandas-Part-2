# Our application will take data from Zillow and our ML application
# Will tell us if it is an apartment or condo to rent/purchase.

# Library aka package we will use for machine learning is sklearn lib and pandas
# Also on this line, we import the tree from this library.
# The Decision Tree is on ype of Classifier
# It is probably the easiest one, and really the only one a human can understand to a point.
import pandas as pd
from sklearn import tree
# Let the user know what our application is about.
print("\n\nOur application will take data from Zillow.com, and then tell you")
print("if it is an apartment or condo for rent/purchase.\n")

# Loading in the data using Pandas
df = pd.read_csv(r"Custom Data Set - Apartment VS Condo.csv")

# Training data
# Features - # of Beds, # of Baths, and Square Footage, and form of purchase in that order
# Form of Purchase: rent = 8, own = 9
# features = [[2, 2, 1316, 9], [2, 2, 1227, 9],[1,1,626, 9],[1, 1, 663, 8],[1, 1, 754, 8], [0, 1, 597, 8]]
# features = df
# # Labels Apartment, Condo
# labels = ["Condo","Condo","Condo","Apartment","Apartment","Apartment"]


# Loading in the data using Pandas
df = pd.read_csv(r"Custom Data Set - Apartment VS Condo.csv")

# Make sure our data is reading in correctly
print("\n______________________________")
print("\t\tTraining Data")
print("______________________________")
print(df)

# removing the first and last columns that aren't important to the bot
labels = list(df['Type of Housing'].values.tolist())
features = df.drop(columns = ['Type of Housing', 'Rent/Own']).values.tolist()

# Training
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)
# # Use our AI ML app to predict what type of housing it will be
# # On homes it has never seen before, based on learning from the training data.
print("\n______________________________")
print("\t\tTesting Data")
print("______________________________")
# This apartment is not in the training data
# This apartment can be found here:
#                       https://www.zillow.com/homedetails/9715-NE-Juanita-Dr-APT-110B-Kirkland-WA-98034/48678412_zpid/
print("This should be an apartment with # of beds: 2, # of baths: 2, SQFT: 1500.")
print(clf.predict([[2,2,1500]]))
# This apartment is not in the training data
# This apartment can be found here:
#                       https://www.zillow.com/homedetails/303-2nd-St-S-APT-B3-Kirkland-WA-98033/2095125720_zpid/
print("This should be a condo with # of beds: 2, # of baths: 2, SQFT: 2007.")
print(clf.predict([[2,2,2007]]))







