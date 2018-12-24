import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scikitplot as skplt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from dataPreprocessing import data_preprocessing, standardize_data
from univariateAnalysis import frequency_measure_visualized, plotting_histograms_for_column_list, plotting_boxplots_for_column_list
from univariateAnalysis import finding_outliers_for_columns_list, plotting_scatter_plot_for_columns, plotting_KDE_plot_for_columns
from univariateAnalysis import explicative_structure_table_with_bins
from explicativeStructureTables import run_EST, explicative_structure_table_multiple_columns
from utils import load_data
from bivariateAnalysis import perform_ols_for_list, print_anova_table, contingency_table_for_list_print_pretty
from LogitRegression_and_plots import perform_logit, descriptive_analysis_of_logit, descriptive_analysis_of_logit_given_dataset
from clusterAnalysis import best_k_for_kmeans_given_data, k_means_analysis_with_silhouette_plotting, hierarchical_cluster_analysis
from sklearn.model_selection import train_test_split


def main():
    path = "TelcoCustomerChurn.csv"
    df_telco = pd.read_csv(path)
    df = data_preprocessing(df_telco)


    column_names_to_drop = ['Churn_Yes', 'customerID', 'TotalCharges']
    columns_needing_bins = ['tenure', 'MonthlyCharges']
    # max values for tenure is 72 and for monthly charges 118.7
    bins_for_columns = [[i for i in range(73)],
                        [i for i in range(120)]]
    column_names = list(df)
    for name in column_names_to_drop:
        column_names.remove(name)
    for name in columns_needing_bins:
        column_names.remove(name)
    # to see the histograms uncomment lines 73 and 74 in file explicativeStructureTables.py
    # this is done for convinience of running the code
    run_EST(column_names, columns_needing_bins, bins_for_columns, df)

    column_names = ['Dependents_Yes', 'Partner_Yes']
    # grouping by column name, counting the occurrences and the churn in category
    print(explicative_structure_table_multiple_columns(column_names, df))

    df = data_preprocessing(load_data(path))
    names_to_drop = ['tenure', 'MonthlyCharges', 'TotalCharges', 'customerID']
    frequency_measure_visualized(df, names_to_drop)

    names_to_plot = ['tenure', 'MonthlyCharges']
    plotting_histograms_for_column_list(names_to_plot, df)
    plotting_boxplots_for_column_list(names_to_plot, df)

    names_for_outliers_detection = ['tenure', 'MonthlyCharges']
    df_without_outliers = finding_outliers_for_columns_list(names_for_outliers_detection, df)

    # plotting the dataset with respect to tenure and monthly charges with churn signified with color.
    names_to_draw = ['tenure', 'MonthlyCharges']
    plotting_scatter_plot_for_columns(names_to_draw, df)

    # visualization for some of the data points
    names_to_draw = ['SeniorCitizen_Yes', 'InternetService_Fiber optic']
    bandwidth = 0.1
    plotting_KDE_plot_for_columns(names_to_draw, df, bandwidth)

    # plotting the churn rate with respect to the bins one by one -> done for both continous variables
    # also this should be run before standardizing the data
    tenure_bin_split = explicative_structure_table_with_bins('tenure', df, [i for i in range(73)])
    plt.scatter([i for i in range(73)], tenure_bin_split.iloc[:, 2], c=(0.11, 0.7, 0.7))
    plt.ylabel("Average churn rate")
    plt.xlabel("Tenure")
    plt.legend(loc='upper right')
    plt.show()

    monthly_bin_split = explicative_structure_table_with_bins('MonthlyCharges', df, [i for i in range(120)])
    plt.scatter([i for i in range(120)], monthly_bin_split.iloc[:, 2], c=(0.11, 0.7, 0.7))
    plt.ylabel("Average Churn Rate")
    plt.xlabel("Monthly charges")
    plt.legend(loc='upper right')
    plt.show()

    # BIVARIATE ANALYSIS
    df = data_preprocessing(load_data(path),standardize=True)
    # The reference variable is Churn, which is qualitative. For this reason we will have to use ANOVA when comparing it
    # with a quantitative variable and the contingency table when comparing it with a qualitative variable.
    column_reference = "Churn_Yes"
    df.rename(index=str,
              columns={'Contract_Month-to-month': 'Contract_Monthly',
                       'PaymentMethod_Electronic check': 'PaymentMethod_Electronic',
                       'InternetService_Fiber optic': 'InternetService_Fiber'},
              inplace=True)
    names_cramer = ['Contract_Monthly',
             'PaymentMethod_Electronic',
             'InternetService_Fiber',
             'SeniorCitizen_Yes']
    
    names_anova = ['tenure',
             'Contract_Monthly']
    
    contingency_table_for_list_print_pretty(df, column_reference, names_cramer) 
    # ANOVA analysis 
    dv_name = column_reference
    iv_names = names_anova
    regression = perform_ols_for_list(df, dv_name, iv_names)
    print_anova_table(regression)
    # Heatmap for the correlation between variables
    sns.heatmap(df.corr())
    plt.show()
    # corelation between monthly charges, total charges and tenure
    df.dropna()
    names = ['tenure', 'MonthlyCharges', 'TotalCharges']
    sns.heatmap(df[names].corr())
    plt.show()

    # CLUSTER ANALYSIS

    df = data_preprocessing(load_data(path), standardize=True)
    columns_for_clustering = ['tenure',
                              'Contract_Month-to-month',
                              'PaymentMethod_Electronic check',
                              'MonthlyCharges',
                              'InternetService_Fiber optic']
    df = df[columns_for_clustering]
    df = standardize_data(df, True, columns_for_clustering)

    # dendrogram for our data
    hierarchical_cluster_analysis(df)

    # using the code from assignment 2 we find the best number of clusters
    best_cluster_number = best_k_for_kmeans_given_data(df)
    k_means_analysis_with_silhouette_plotting(df, best_cluster_number)
    print("best number of clusters")

    # Running Logit

    # Loading and preparing the data
    df = pd.read_csv("TelcoCustomerChurn.csv")
    df = data_preprocessing(df, standardize=True)

    # Filtering and fitting variables that should be included
    logit_dv = 'Churn_Yes'
    # adding an intercept
    df['intercept'] = 1.0
    # calculating the powers of continous variables to capture their change
    df["tenure2"] = df["tenure"] ** 2
    df["MonthlyCharges2"] = df["MonthlyCharges"] ** 2
    df["MonthlyCharges3"] = df["MonthlyCharges"] ** 3

    # selecting desired columns
    logit_ivs = [
        'tenure',
        'tenure2',
        'MonthlyCharges',
        'SeniorCitizen_Yes',
        'InternetService_Fiber optic',
        'Contract_Month-to-month',
        'PaymentMethod_Electronic check',
        'intercept'
    ]
    x_train, x_test, y_train, y_test = train_test_split(df[logit_ivs], df[logit_dv], test_size=0.30, random_state=42)
    # =========================================ACCURACY OF THE MODEL===================================================
    logRegress = LogisticRegression()
    logRegress.fit(x_train, y_train)
    accuracy = logRegress.score(x_train, y_train)
    # =================================================================================================================
    print(accuracy)
    logit = sm.Logit(y_train, x_train)
    logit_result = logit.fit()

    descriptive_analysis_of_logit_given_dataset(logit_result, y_test, x_test, df)    # running GLM
    gamma_model = sm.GLM(y_train, x_train, family=sm.families.Gamma())
    gamma_results = gamma_model.fit()
    print(gamma_results.summary())


main()
