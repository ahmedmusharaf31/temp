import pandas as pd # Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


df=sns.load_dataset("titanic").copy()  # Copy the dataset to avoid chained assignment issues
print("Dataset Info:")
print(df.info())


print("Dataset Head:")
print(df.head())  # Display first 5 rows


df.drop(columns=['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male'], inplace=True, errors='ignore')
df['age']=df['age'].fillna(df['age'].median())  # Fill missing Age with median
df['embarked']=df['embarked'].fillna(df['embarked'].mode()[0])  # Fill missing Embarked with mode
df['fare']=df['fare'].fillna(df['fare'].median())  # Fill missing Fare with median


label_enc=LabelEncoder()
df['sex']=label_enc.fit_transform(df['sex'])  # Male=1, Female=0
df['embarked']=label_enc.fit_transform(df['embarked'])  # C=0, Q=1, S=2



df=df.dropna(subset=['survived']) # Drop rows with missing target values


X=df.drop(columns=['survived'])  # Features
y=df['survived']  # Target variable
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)



model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)  # Training the Decision Tree Model here
model.fit(X_train, y_train)


y_pred=model.predict(X_test)  # Evaluating the model


accuracy=accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)  # Checking accuracy score


# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))


# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Plotting Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=X.columns, class_names=['Not Survived', 'Survived'], filled=True)
plt.show()
