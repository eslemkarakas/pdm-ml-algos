# import standard packages
from scipy.stats import ttest_ind, f_oneway, chi2_contingency

class HypothesisTester:
    def __init__(self, df):
        self.df = df

    def independent_t_test(self, target_column, evaluated_column, alpha=0.05):
        first_group = self.df[self.df[evaluated_column].notna()][target_column]
        second_group = self.df[self.df[evaluated_column].isna()][target_column]

        stat, p_value = ttest_ind(first_group, second_group, equal_var=False)

        return p_value < alpha
    
    def one_way_anova(self, target_column, grouping_column, alpha=0.05):
        groups = [group_data[target_column] for _, group_data in self.df.groupby(grouping_column)]

        stat, p_value = f_oneway(*groups)

        return p_value < alpha

    def chi_square_test(self, observed_column, expected_column, alpha=0.05):
        observed = self.df[observed_column]
        expected = self.df[expected_column]

        stat, p_value, _, _ = chi2_contingency([observed, expected])

        return p_value < alpha