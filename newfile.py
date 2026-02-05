import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import math

# Step 1: Construct dataset
wine_data = pd.DataFrame({
    "Price": ["m","e","e","m","m","m","e","b","b","m"],
    "Type": ["w","d","d","w","w","d","sp","r","d","d"],
    "Taste": ["a","t","a","sw","sw","t","a","t","a","b"],
    "Origin": ["aaa","am","am","am","eu","eu","aaa","eu","aaa","aaa"],
    "Decision": ["to buy","to buy","to buy","to buy","to buy",
                 "not to buy","not to buy","not to buy","not to buy","not to buy"]
})

# Step 2: Define entropy calculation
def calculate_entropy(column):
    class_counts = column.value_counts()
    total = len(column)
    entropy_value = 0

    for count in class_counts:
        probability = count / total
        entropy_value -= probability * math.log2(probability)

    return entropy_value

# Step 3: Define information gain calculation
def calculate_ig(dataset, feature, target):
    base_entropy = calculate_entropy(dataset[target])
    feature_values = dataset[feature].unique()

    feature_entropy = 0
    for val in feature_values:
        subset = dataset[dataset[feature] == val]
        weight = len(subset) / len(dataset)
        feature_entropy += weight * calculate_entropy(subset[target])

    return base_entropy - feature_entropy

# Step 4: Parent entropy
print("\nParent Entropy Details")
print("--------------------------")
print("Number of records:", len(wine_data))
print("Decision distribution:")
print(wine_data["Decision"].value_counts())
print("Parent entropy =", calculate_entropy(wine_data["Decision"]))

# Step 5: Information Gain for Type
ig_type = calculate_ig(wine_data, "Type", "Decision")

print("\nInformation Gain for Type")
print("--------------------------")
print("Distinct Type values:", wine_data["Type"].unique())
print("IG(Type) =", ig_type)

# Step 6: Encode categorical attributes
encoded_data = wine_data.copy()
label_enc = LabelEncoder()

for col in encoded_data.columns:
    encoded_data[col] = label_enc.fit_transform(encoded_data[col])

X_data = encoded_data.drop("Decision", axis=1)
y_data = encoded_data["Decision"]

# Step 7: Train Decision Tree
dt_model = DecisionTreeClassifier(criterion="entropy", random_state=0)
dt_model.fit(X_data, y_data)

# Step 8: Identify root feature
root_node_index = dt_model.tree_.feature[0]
print("\nDecision Tree Root Feature")
print("--------------------------")
print("Root split attribute:", X_data.columns[root_node_index])

# Step 9: Information gain for other attributes
print("\nInformation Gain of Remaining Attributes")
print("------------------------------------------")
print("IG(Price) =", calculate_ig(wine_data, "Price", "Decision"))
print("IG(Taste) =", calculate_ig(wine_data, "Taste", "Decision"))
print("IG(Origin) =", calculate_ig(wine_data, "Origin", "Decision"))
