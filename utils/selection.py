import pandas as pd
from scipy.stats import chi2_contingency

def chi_square_test(y_train, X_train, feature):
    
    contingency_table = pd.crosstab(y_train, X_train[feature])
    p_value = chi2_contingency(contingency_table)[1]
    
    return p_value