import pandas as pd
pd.set_option('display.max_columns',10)
from dataPreprocessing import data_preprocessing
from utils import load_data

def explicative_structure_table(column_name, df):
    churn_yes = "Churn_Yes"
    percentage_population = "Percentage population"
    average_churn_rage = "Average churn rate"
    n = len(df)
    # grouping by column name, counting the occurences and the churn in each group
    table = df.groupby(column_name).agg({column_name: 'count', churn_yes: 'sum'})
    # calculate the percentage of population belonging to each group
    table[percentage_population] = round((table[column_name] / n), 5)
    table[average_churn_rage] = table[churn_yes] / table[column_name]
    table = table.drop([column_name, churn_yes], axis=1)
    table = table.append({percentage_population: sum(table[percentage_population]),
                          average_churn_rage: sum(df[churn_yes] / n)}, ignore_index=True)
    table.insert(0, column_name, ['True', 'False', 'Overall Population'])
    return table


# TO BE CHECKED
def explicative_structure_table_multiple_columns(column_names, df):
    churn_yes = "Churn_Yes"
    # grouping by column name, counting the occurences and the churn in each group
    table = df.groupby(column_names)[churn_yes].sum()
    # it requires a double manual check.
    for i in range(len(table.index.levels)):
        for j in range(len(table.index.levels[i])):
            number_of_occurences = len(df.loc[(df[table.index.levels[0].name] == i) &
                                              (df[table.index.levels[1].name] == j)])
            table[i][j] = table[i][j] / number_of_occurences
            print(f'{number_of_occurences} for {table.index.levels[0].name} equal to {i} and for {table.index.levels[1].name} equal to {j}')
    table = table.unstack()
    return table


def explicative_structure_table_with_bins(column_name, df, bin_division):
    churn_yes = "Churn_Yes"
    percentage_population = "Percentage population"
    average_churn_rage = "Average churn rate"
    n = len(df)
    new_c_name = column_name + "_bins"
    df[new_c_name] = pd.cut(df[column_name], bin_division)
    # grouping by column name, counting the occurences and the churn in each group
    table = df.groupby(new_c_name).agg({new_c_name: 'count', churn_yes: 'sum'})
    # calculate the percentage of population belonging to each group
    table[percentage_population] = round((table[new_c_name] / n), 5)
    table[average_churn_rage] = table[churn_yes] / table[new_c_name]
    table = table.drop([new_c_name, churn_yes], axis=1)
    table = table.append({percentage_population: sum(table[percentage_population]),
                          average_churn_rage: sum(df[churn_yes] / n)}, ignore_index=True)
    custom_labels = ranges_string_creation(bin_division)
    custom_labels.append("Overall Population")
    table.insert(0, column_name, custom_labels)
    return table


def ranges_string_creation(range_list):
    range_in_string = []
    for i in range(len(range_list)-1):
        range_in_string.append("(" + str(range_list[i]) + ", " + str(range_list[i+1]) + "]")
    return range_in_string


def run_EST(column_list, columns_needing_bins, bins, df):
    for column in column_list:
        print("creating explicative structure table for {}", column)
        print(explicative_structure_table(column, df))

    for i in range(len(columns_needing_bins)):
        print("Explicative structure table for ", columns_needing_bins[i])
        print(explicative_structure_table_with_bins(columns_needing_bins[i], df, bins[i]))


# TESTING
path = "TelcoCustomerChurn.csv"
df = data_preprocessing(load_data(path))
column_names_to_drop = ['Churn_Yes', 'customerID']
columns_needing_bins = ['tenure', 'MonthlyCharges', 'TotalCharges']
# [0, 3, 10, 20, 35, 50, 65, 72]
bins_for_columns = [[i for i in range(73)],
                    [18, 28, 48, 68, 108, 119],
                    [18, 100, 300, 1000, 5000, 9000]]
column_names = list(df)
for name in column_names_to_drop:
    column_names.remove(name)
# for this to run correctly we need to comment out the standardization in data preprocessing.
for name in columns_needing_bins:
    column_names.remove(name)
run_EST(column_names, columns_needing_bins, bins_for_columns, df)

column_names = ['Dependents_Yes', 'Partner_Yes']
# grouping by column name, counting the occurences and the churn in category
print(explicative_structure_table_multiple_columns(column_names, df))
