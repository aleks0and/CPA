import matplotlib.pyplot as plt
import pandas as pd
from utils import load_data
from dataPreprocessing import data_preprocessing, finding_outliers_for_columns_list
from explicativeStructureTables import explicative_structure_table_with_bins

# this has to be corrected as we dont want to be manually adding senior citizen.
def frequency_measure_visualized(df, names_to_drop):
    names_to_drop = set(names_to_drop)
    dummies_names = [x for x in list(df) if x not in names_to_drop]
    plt.subplots_adjust(wspace=2, hspace=2)
    plt.figure(figsize=(10, 15))
    for i in range(len(dummies_names)):
        size = df[dummies_names[i]].value_counts()
        plt.subplot(7, 3, i + 1)
        plt.pie(size, labels=["no", "yes"], shadow=True, autopct='%1.0f%%')
        plt.title(dummies_names[i])
    plt.show()


def plotting_histograms_for_column_list(names_list, df):
    i = 0
    plt.figure(figsize=(10,15))
    plt.subplots_adjust(wspace=1, hspace=1)
    for name in names_list:
        plt.subplot(len(names_list), len(names_list)/2, i+1)
        plt.hist(df[name])
        plt.title(name)
        i += 1
    plt.show()


def plotting_boxplots_for_column_list(names_list, df):
    i = 0
    plt.figure(figsize=(10, 15))
    plt.subplots_adjust(wspace=1, hspace=1)
    for name in names_list:
        plt.subplot(len(names_list), len(names_list)/2, i+1)
        plt.boxplot(df[name])
        plt.title(name)
        i += 1
    plt.show()


def plotting_scatter_plot_for_columns(names_list,df):
    names_list.append('Churn_Yes')
    data_for_scatter = df[names_list]
    data_for_scatter.plot.scatter(x=names_list[0], y=names_list[1], c=names_list[2],
                                  colormap='viridis')
    plt.show()


def plotting_KDE_plot_for_columns(names_list, df, bandwidth):
    data_for_KDE = df[names_list]
    data_for_KDE.plot.kde(bw_method=bandwidth)
    plt.show()

#
# path = "TelcoCustomerChurn.csv"
# df = data_preprocessing(load_data(path))
# names_to_drop = ['tenure', 'MonthlyCharges', 'TotalCharges', 'customerID']
# frequency_measure_visualized(df, names_to_drop)

# names_to_plot = ['tenure', 'MonthlyCharges', 'TotalCharges']
# plotting_histograms_for_column_list(names_to_plot, df)
# plotting_boxplots_for_column_list(names_to_plot, df)
#
# names_for_outliers_detection = ['tenure', 'MonthlyCharges', 'TotalCharges']
# df_without_outliers = finding_outliers_for_columns_list(names_for_outliers_detection, df)
#
# # Works best for continuous variables -> I also added churn yes as the color for markers
# names_to_draw=['tenure', 'MonthlyCharges']
# plotting_scatter_plot_for_columns(names_to_draw, df)
#
# # Try with categorical but most likely we can see the last 4 obs plotted.
# # names_to_draw2=['Dependents_Yes', 'Partner_Yes']
# # plotting_scatter_plot_for_columns(names_to_draw2, df)
#
# # Visualization for the density for categorical variables
# # does the same work as pie charts but looks a bit different, in this case
# # bandwidth changes the width of the curves -> better to keep it low
# names_to_draw = ['Partner_Yes', 'Dependents_Yes']
# bandwidth = 0.1
# plotting_KDE_plot_for_columns(names_to_draw, df, bandwidth)

# plotting the churn rate with respect to the bins one by one -> done for both continous variables
# also this should be run before standardizing the data
# tenure_bin_split = explicative_structure_table_with_bins('tenure', df, [i for i in range(73)])
# plt.scatter([i for i in range(73)], tenure_bin_split.iloc[:, 2], c=(0.11, 0.7, 0.7))
# plt.xlabel("Average churn rate")
# plt.ylabel("Tenure")
# plt.legend(loc='upper right')
# plt.show()
#
# monthly_bin_split = explicative_structure_table_with_bins('MonthlyCharges', df, [i for i in range(120)])
# plt.scatter(monthly_bin_split.iloc[:, 2], [i for i in range(120)], c=(0.11, 0.7, 0.7))
# plt.xlabel("Average Churn Rate")
# plt.ylabel("Monthly charges")
# plt.legend(loc='upper right')
# plt.show()
