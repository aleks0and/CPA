import pandas as pd
pd.set_option('display.max_columns',10)
from dataPreprocessing import data_preprocessing

def explicative_structure_table(column_name, df):
    churn_yes = "Churn_Yes"
    percentage_population = "Percentage population"
    average_churn_rage = "Average churn rate"
    n = len(df)
    # grouping by column name, counting the occurences and the churn in each group
    table = df.groupby(column_name).agg({column_name: 'count', churn_yes: 'sum'})
    # calculate the percentage of population belonging to each group
    table[percentage_population] = round((table[column_name] / n), 3)
    table[average_churn_rage] = table[churn_yes] / table[column_name]
    table = table.drop([column_name, churn_yes], axis=1)
    table = table.append({percentage_population: sum(table[percentage_population]),
                          average_churn_rage: sum(df[churn_yes] / n)}, ignore_index=True)
    table.insert(0, column_name, ['True', 'False', 'Overall Population'])
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
    table[percentage_population] = round((table[new_c_name] / n), 3)
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


def run_EST_for_dummies_list(columnList, df):
    for column in columnList:
        print("creating explicative structure table for {}", column)
        print(explicative_structure_table(column, df))


#TESTING

path = "TelcoCustomerChurn.csv"
df_telco = pd.read_csv(path)
df = data_preprocessing(df_telco)
column_names_to_drop = ['Churn_Yes', 'customerID']
columns_needing_bins = ['tenure', 'MonthlyCharges', 'TotalCharges']
bins_for_columns = [[0, 3, 10, 20, 35, 50, 65, 72],
                    [18, 28, 48, 68, 108, 119],
                    [18, 100, 300, 1000, 5000, 9000]]
column_names = list(df)
for name in column_names_to_drop:
    column_names.remove(name)
for name in columns_needing_bins:
    column_names.remove(name)
for i in range(len(columns_needing_bins)):
    print("Explicative structure table for ",columns_needing_bins[i])
    print(explicative_structure_table_with_bins(columns_needing_bins[i], df, bins_for_columns[i]))
run_EST_for_dummies_list(column_names,df)