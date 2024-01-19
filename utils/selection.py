# import standard packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=3,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    random_state=0,
    shuffle=False,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
feature_names = [f"feature {i}" for i in range(X.shape[1])]

class ImpurityBasedFeatureSelection():
    def __init__(self):
        self.features = None
        self.feature_importances = None
        self.selector = RandomForestClassifier(random_state=0)
        self.std = None
        self.forest_importances = None
        
    def fit(self, X_train, y_train):
        """Fit the RandomForestClassifier on the training data."""
        self.selector.fit(X_train, y_train)
        
    def find_importances(self):
        """Calculate feature importances."""
        self.feature_importances = self.selector.feature_importances_
    
    def calculate_forest_importances(self, features):
        """Calculate and return forest importances."""
        self.std = np.std([tree.feature_importances_ for tree in self.selector.estimators_], axis=0)
        self.forest_importances = pd.Series(self.feature_importances, index=features)
        return self.forest_importances
    
    def select_features_above_threshold(self, threshold=0.05):
        """Select features with importances above a specified threshold."""
        selected_features = self.forest_importances[self.forest_importances >= threshold]
        return selected_features
    
    def plot(self, show_error_bars=True):
        """Plot feature importances with optional error bars."""
        fig, ax = plt.subplots()
        
        if show_error_bars:
            ax.bar(self.forest_importances.index, self.forest_importances, yerr=self.std)
        else:
            ax.bar(self.forest_importances.index, self.forest_importances)
            
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        plt.show();
        
clf = ImpurityBasedFeatureSelection()
clf.fit(X_train, y_train)
clf.find_importances()
forest_importances = clf.calculate_forest_importances(feature_names)
print(clf.select_features_above_threshold())

clf.plot()