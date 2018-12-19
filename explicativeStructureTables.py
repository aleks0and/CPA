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


path = "TelcoCustomerChurn.csv"
df_telco = pd.read_csv(path)
df = data_preprocessing(df_telco)
print(explicative_structure_table('gender_Male', df))
