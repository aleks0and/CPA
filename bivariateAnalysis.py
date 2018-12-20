import statsmodels.api as sm
import scipy as sci
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from utils import load_data
from dataPreprocessing import data_preprocessing

pd.set_option('display.max_columns',10)


def contingency_table_for_list_print_ugly(df, col_reference, col_list):
    for name in col_list:
        table = sm.stats.Table.from_data(df[[col_reference,name]])
        print(table.table)


def cramers_V(matrix):
    chi2 = sci.stats.chi2_contingency(matrix)[0]
    n = matrix.sum().sum()
    phi2 = chi2/n
    r, k = matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


def contingency_table_for_list_print_pretty(df, col_reference, col_list):
    for name in col_list:
        contingency_table = pd.crosstab(index= df[col_reference], columns=df[name])
        print(contingency_table)
        cramer = cramers_V(contingency_table)
        print(f"cramer's V: {cramer}\n")


# this function adds the effect size (eta sqared)
def print_anova_table(aov):
    aov['mean_sq'] = aov[:]['sum_sq'] / aov[:]['df']
    aov['eta_sq'] = aov[:-1]['sum_sq'] / sum(aov['sum_sq'])
    aov['omega_sq'] = (aov[:-1]['sum_sq'] - (aov[:-1]['df'] * aov['mean_sq'][-1])) / \
                      (sum(aov['sum_sq']) + aov['mean_sq'][-1])
    cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
    aov = aov[cols]
    print(aov)


def perform_ols_for_list(df, dv_name, iv_names):
    equation = dv_name + ' ~'
    for name in iv_names:
        equation += ' ' + name + ' +'
    equation = equation[:-2]
    print(equation)
    regression = ols(formula=equation, data=df).fit()
    # type 2 ANOVA
    return sm.stats.anova_lm(regression, typ=2)


path = "TelcoCustomerChurn.csv"
df = data_preprocessing(load_data(path))
# The reference variable is Churn, which is qualitative. For this reason we will have to use ANOVA when comparing it
# with a quantitative variable and the contingency table when comparing it with a qualitative variable.
column_reference = "Churn_Yes"
# here just insert the names you want to drop from the overall columns and it will do its magic
# names_to_drop = ['tenure', 'MonthlyCharges', 'TotalCharges', 'customerID', 'Churn_Yes']
# names_to_drop = set(names_to_drop)
# names = [x for x in list(df) if x not in names_to_drop]
# in other case just insert the columns you want tables for below.
names = ["SeniorCitizen_Yes"]
contingency_table_for_list_print_pretty(df, column_reference, names)
contingency_table_for_list_print_ugly(df, column_reference, names)
# ANOVA analysis
dv_name = column_reference
iv_names = ['tenure', 'MonthlyCharges', 'TotalCharges']
regression = perform_ols_for_list(df, dv_name, iv_names)
print_anova_table(regression)
