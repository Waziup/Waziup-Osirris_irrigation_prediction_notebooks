# -*- coding: utf-8 -*-
"""
Created on Wed May  3 10:54:49 2023

@author: felix
"""

#TODO:general names in csv export

import pycaret 
from pycaret.time_series import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno 
import sys
from sklearn.impute import KNNImputer

# global variables
csv_filepath = "binned_removed.csv"
fh = 48



# Load from CSV file
def load_data(path):
    # creating a data frame
    data = pd.read_csv("binned_removed.csv",header=0)
    print(data.head())
    return data


# Impute missing data & apply rolling mean (imputation & cleaning)
def fill_gaps(data):
    # Show if there are any missing values inside the data
    print("This is before: \n",data.isna().any())
    msno.matrix(data);
    
    # Show heatmap
    msno.heatmap(data);
    
    # Show Dedrogram
    msno.dendrogram(data)
    
    # Copy the data
    data = data.copy(deep=True)
    
    # Init the transformer
    knn_imp = KNNImputer(n_neighbors=10)
    
    # Fit/transform
    data.iloc[:,1:8] = knn_imp.fit_transform(data.iloc[:,1:8])
    
    # Plot correlation heatmap of missingness
    msno.matrix(data)
    
    # Show if there are any missing values inside the data
    data.isna().any()
    
    # Plot before rolling mean
    data.plot(subplots=True,figsize=(20,50))
    
    # Apply rolling mean
    for col in data:
        if col == 'Time':
            #print('This is Time')
            continue
        else:
            data[col] = data[col].rolling(window=15, win_type='gaussian').mean(std=15)
    
    # Plot after rolling mean
    data.plot(subplots=True,figsize=(20,50))
    
    return data


# Augment the dataset creating new features
def create_features(data):
    # Create average cols => needs to be adopted
    data['grouped_soil'] = data[['638de53168f31919048f189d, 63932c6e68f319085df375b5; B2_solar_x2_03, Soil_tension', 
                                 '638de57568f31919048f189f, 63932c3668f319085df375ab; B4_solar_x1_03, Soil_tension'
                                 ]].mean(axis=1)
    
    data['grouped_resistance'] = data[['638de53168f31919048f189d, 63932c6e68f319085df375b7; B2_solar_x2_03, Resistance', 
                                       '638de56a68f31919048f189e, 63932ce468f319085df375cb; B3_solar_x1_02, Resistance', 
                                       '638de57568f31919048f189f, 63932c3668f319085df375ad; B4_solar_x1_03, Resistance'
                                       ]].mean(axis=1)
    
    data['grouped_soil_temp'] = data[['638de53168f31919048f189d, 63932c6e68f319085df375b9; B2_solar_x2_03, Soil_temperature', 
                                      '638de57568f31919048f189f, 63932c3668f319085df375af; B4_solar_x1_03, Soil_temperature'
                                      ]].mean(axis=1)
    
    # Create rolling mean: introduces NaN again -> later just cut off
    data['rolling_mean_grouped_soil'] = data['grouped_soil'].rolling(window=5, win_type='gaussian').mean(std=5)
    data['rolling_mean_grouped_soil_temp'] = data['grouped_soil_temp'].rolling(window=5, win_type='gaussian').mean(std=5)
    
    # Create time related features
    data['hour'] = data['Time'].dt.hour
    data['minute'] = data['Time'].dt.minute
    data['date'] = data['Time'].dt.day
    data['month'] = data['Time'].dt.month
    
    # bring back to order -> not important
    data = data[['Time', 'hour', 'minute', 'date', 'month', 'grouped_soil', 
                 'grouped_resistance', 'grouped_soil_temp', 'rolling_mean_grouped_soil', 
                 'rolling_mean_grouped_soil_temp', 
                 '638de53168f31919048f189d, 63932c6e68f319085df375b5; B2_solar_x2_03, Soil_tension', 
                 '638de53168f31919048f189d, 63932c6e68f319085df375b7; B2_solar_x2_03, Resistance', 
                 '638de53168f31919048f189d, 63932c6e68f319085df375b9; B2_solar_x2_03, Soil_temperature', 
                 '638de56a68f31919048f189e, 63932ce468f319085df375cb; B3_solar_x1_02, Resistance', 
                 '638de57568f31919048f189f, 63932c3668f319085df375ab; B4_solar_x1_03, Soil_tension', 
                 '638de57568f31919048f189f, 63932c3668f319085df375ad; B4_solar_x1_03, Resistance', 
                 '638de57568f31919048f189f, 63932c3668f319085df375af; B4_solar_x1_03, Soil_temperature'
                 ]]

    # drop those values without rolling_mean
    data = data[18:]
    
    # reset changed index(due to drop)
    data = data.reset_index()
    
    data = data.drop(['index'],axis=1)
    
    return data

# Normalize the data in min - max approach from 0 - 1
def normalize(data):
    # feature scaling
    data.describe()
    
    # Min-Max Normalization
    df = data.drop(['Time','rolling_mean_grouped_soil', 'hour', 'minute', 'date', 'month'], axis=1)
    df_norm = (df-df.min())/(df.max()-df.min())
    df_norm = pd.concat([df_norm, data['Time'],data['hour'], data['minute'], data['date'], data['month'], data.rolling_mean_grouped_soil], 1)

    # bring back to order -> not important
    data = data[['Time', 'hour', 'minute', 'date', 'month', 'grouped_soil', 
                 'grouped_resistance', 'grouped_soil_temp', 'rolling_mean_grouped_soil', 
                 'rolling_mean_grouped_soil_temp', 
                 '638de53168f31919048f189d, 63932c6e68f319085df375b5; B2_solar_x2_03, Soil_tension', 
                 '638de53168f31919048f189d, 63932c6e68f319085df375b7; B2_solar_x2_03, Resistance', 
                 '638de53168f31919048f189d, 63932c6e68f319085df375b9; B2_solar_x2_03, Soil_temperature', 
                 '638de56a68f31919048f189e, 63932ce468f319085df375cb; B3_solar_x1_02, Resistance', 
                 '638de57568f31919048f189f, 63932c3668f319085df375ab; B4_solar_x1_03, Soil_tension', 
                 '638de57568f31919048f189f, 63932c3668f319085df375ad; B4_solar_x1_03, Resistance', 
                 '638de57568f31919048f189f, 63932c3668f319085df375af; B4_solar_x1_03, Soil_temperature'
                 ]]

    return df_norm


# Split dataset into train and test set
def split_data(data, split_date):
    return data[data['Time'] <= split_date].copy(), \
           data[data['Time'] >  split_date].copy()
           
           
# Delete the ones that are non consecutive
def delete_nonconsecutive_rows(df, column_name, min_consecutive):
    arr = df[column_name].to_numpy()
    i = 0
    while i < len(arr) - 1:
        if arr[i+1] == arr[i] + 1:
            start_index = i
            while i < len(arr) - 1 and arr[i+1] == arr[i] + 1:
                i += 1
            end_index = i
            if end_index - start_index + 1 < min_consecutive:
                df = df.drop(range(start_index, end_index+1))
        i += 1
    return df


# Create visual representation of irrigation times
def highlight(data_plot, ax, neg_slope):
    for index, row in neg_slope.iterrows():
        current_index = int(row['index'])
        #print(current_index)
        ax.axvspan(current_index-10, current_index+10, facecolor='pink', edgecolor='none', alpha=.5)
    
    
# Create ranges to remove from data
def create_split_tuples(df, indices_to_omit):
    # Sort the indices in ascending order
    indices_to_omit = sorted(indices_to_omit)

    # Create a list of index ranges to remove
    ranges_to_remove = []
    start_idx = None
    for idx in indices_to_omit:
        if start_idx is None:
            start_idx = idx
        elif idx == start_idx + 1:
            start_idx = idx
        else:
            ranges_to_remove.append((int(start_idx), int(idx-1)))
            start_idx = idx
    if start_idx is not None:
        ranges_to_remove.append((int(start_idx), df.index.max()))
        
    print("Irrigation times to be omitted: ", ranges_to_remove)
    print("type: ", type(ranges_to_remove[0][0]))

    return ranges_to_remove


# Split data to split dataframes
def split_dataframe(df, index_ranges):
    dfs = []
    for i, (start, end) in enumerate(index_ranges):
        if index_ranges[i][1]-index_ranges[i][0] < 50:
            continue
        else:
            dfs.append(df.iloc[index_ranges[i][0]:index_ranges[i][1]])
            
    return dfs

# Main function to split dataframes
def split_sub_dfs(data, data_plot):
    # calculate slope of "rolling_mean_grouped_soil"
    f = data.rolling_mean_grouped_soil
    data['gradient'] = np.gradient(f)
    
    # create dataframe with downward slope
    neg_slope = pd.DataFrame({"index":[],
                             "rolling_mean_grouped_soil":[],
                             "gradient":[]}
                            )
    
    for index, row in data.iterrows():
        if row['gradient'] < -0.07: 
            #print(index, row['rolling_mean_grouped_soil'], row['gradient'])
            current_series = pd.Series([int(round(index,0)), row['rolling_mean_grouped_soil'],
                                        row['gradient']], index=['index', 
                                                                 'rolling_mean_grouped_soil', 
                                                                 'gradient']).to_frame().T
            neg_slope = neg_slope.append(current_series)
    
    # dont ask, I love pandas^^
    neg_slope_2 = pd.DataFrame({'index':[], 'rolling_mean_grouped_soil':[], 'gradient': []})
    neg_slope_2 = pd.concat([neg_slope_2, neg_slope], ignore_index=True)
    neg_slope = neg_slope_2
    
    # Delete the ones that are non consecutive
    neg_slope = delete_nonconsecutive_rows(neg_slope, 'index', 5)
    with open('output.txt', 'w') as f:
        print(neg_slope, file=f)
    
    # visualize areas with downward slope
    ax = data_plot.drop(['Time'], axis=1).plot()
    highlight(data_plot, ax, neg_slope)
    ax.figure.suptitle("""Irrigation times highlighted\n\n""", fontweight ="bold") 
    ax.figure.savefig('irrigation_times_temp.png', dpi=400)
    
    # convert to numpy array and to int
    neg_slope_indices = neg_slope['index'].to_numpy()
    neg_slope_indices = neg_slope_indices.astype(np.int32)
    
    # Create ranges to remove from data
    tuples_to_remove = create_split_tuples(data, neg_slope_indices)
    
    # Split data to split dataframes
    sub_dfs = split_dataframe(data, tuples_to_remove) 
    
    # print dataframes
    with open('output.txt', 'a') as f:
        print("There are ", len(sub_dfs), " dataframes now.", file=f)
        for sub_df in sub_dfs:
            print(sub_df.head(1), file=f)
            print(len(sub_df), file=f)
            sub_df.drop(['Time', 'hour', 'minute', 'date', 'month'], axis=1).plot()
            
    return data, sub_dfs


# Find global max and min in all sub_dfs and cut them from min to max 
# => train data will start with min and end with max 
def format_begin_end(sub_dfs):
    cut_sub_dfs = []
    for i in range(len(sub_dfs)):
        # reset "new" index
        sub_dfs[i] = sub_dfs[i].reset_index()
        
        # index
        global_min_index = sub_dfs[i]['rolling_mean_grouped_soil'].idxmin()
        global_max_index = sub_dfs[i]['rolling_mean_grouped_soil'].idxmax()
        # value
        global_min = sub_dfs[i]['rolling_mean_grouped_soil'].min()
        global_max = sub_dfs[i]['rolling_mean_grouped_soil'].max()
        
        print(i,": ",global_min_index, "value:", global_min, global_max_index, "value:", global_max, "length:", global_max_index-global_min_index)
        print(i,": ",global_min_index, "value:", sub_dfs[i]['rolling_mean_grouped_soil'][global_min_index], global_max_index, "value:", sub_dfs[i]['rolling_mean_grouped_soil'][global_max_index], "length:", global_max_index-global_min_index)
        
        
        cut_sub_dfs.append(sub_dfs[i].iloc[global_min_index:global_max_index])
    
    # Print them    
    for df in cut_sub_dfs:
        df.drop(['index','Time','hour', 'minute', 'date', 'month'],axis=1).plot()
        
    # Preserve old index and clean
    for i in range(len(cut_sub_dfs)):
        cut_sub_dfs[i] = cut_sub_dfs[i].reset_index()
        # clean dataframe
        cut_sub_dfs[i] = cut_sub_dfs[i].drop(['level_0'], axis=1)
        cut_sub_dfs[i] = cut_sub_dfs[i].rename(columns={'index':'orig_index'})
        
    # Print head of dfs
    i = 1
    with open('output.txt', 'a') as f:
        for df in cut_sub_dfs:
            print("Dataframe: ", i, file=f)
            i+=1
            print(df.iloc[:1], file=f)
        
    return cut_sub_dfs


# Combine them to one dataframe
def combine_dfs(cut_sub_dfs):
    # save all dataframes to one and rename
    df_comb = pd.DataFrame()
    for i in range(len(cut_sub_dfs)):
        # copy elements to one df_comb
        
        df_comb = pd.concat([df_comb, cut_sub_dfs[i]['orig_index']], axis=1)
        df_comb = pd.concat([df_comb, cut_sub_dfs[i]['rolling_mean_grouped_soil']], axis=1)
        
        df_comb = df_comb.rename(columns={'orig_index':'orig_index_' + str(i)})
        df_comb = df_comb.rename(columns={'rolling_mean_grouped_soil':'rolling_mean_grouped_soil_' + str(i)})
    
    # series are not of same length => visualized here!
    df_comb.drop(['orig_index_0', 'orig_index_1', 'orig_index_2', 'orig_index_3'],axis=1).plot()
    
    return df_comb


# Data preparation pipeline, calls other subfunction to perform the task
def prepare_data(path):
    # Load data from file
    data = load_data(path)
    
    # Impute gaps in data
    data = fill_gaps(data)
    
    # cut timezones from time string to convert to datetime64
    data['Time'] = data['Time'].str[:-9]
    data['Time'] = pd.to_datetime(data['Time'])
    
    # create additional features
    data = create_features(data)
    
    # Normalization
    data = normalize(data)
    print(data.iloc[0])
    
    print(data.head(0))
    
    # Plot important data
    data_plot = data.drop(['hour','minute','date','month','grouped_soil',
                           'grouped_resistance','grouped_soil_temp',
                           '638de53168f31919048f189d, 63932c6e68f319085df375b5; B2_solar_x2_03, Soil_tension',
                           '638de53168f31919048f189d, 63932c6e68f319085df375b7; B2_solar_x2_03, Resistance',
                           '638de53168f31919048f189d, 63932c6e68f319085df375b9; B2_solar_x2_03, Soil_temperature',
                           '638de56a68f31919048f189e, 63932ce468f319085df375cb; B3_solar_x1_02, Resistance',
                           '638de57568f31919048f189f, 63932c3668f319085df375ab; B4_solar_x1_03, Soil_tension'
                           ,'638de57568f31919048f189f, 63932c3668f319085df375ad; B4_solar_x1_03, Resistance',
                           '638de57568f31919048f189f, 63932c3668f319085df375af; B4_solar_x1_03, Soil_temperature'
                           ], axis=1)
    
    data_plot.set_index('Time').plot(figsize = (20,10))
    
    # Split data in test and train and show -> not needed any more
    train, test = split_data(data, pd.to_datetime('2023-03-08 15:30')) # splitting the data for training before 15th June
    plt.figure(figsize=(20,10))
    plt.xlabel('Time')
    plt.ylabel('rolling_mean_grouped_soil')
    plt.plot(train.index,train['rolling_mean_grouped_soil'],label='train')
    plt.plot(test.index,test['rolling_mean_grouped_soil'],label='test')
    plt.legend()
    plt.show()
    
    # Split according to irrigation times
    data, sub_dfs = split_sub_dfs(data, data_plot)
    
    # Find global max and min in all sub_dfs and cut them from min to max 
    cut_sub_dfs = format_begin_end(sub_dfs)
    
    # Combine all cut_sub_dfs to one df_comb
    df_comb = combine_dfs(cut_sub_dfs)
    
    return data, data_plot, df_comb, cut_sub_dfs

# Create model in pycaret
def create_and_compare_model(cut_sub_dfs):
    
    # call setup of pycaret
    exp=[]
    for i in range(len(cut_sub_dfs)):
        exp.append(TSForecastingExperiment())
        
        # check the type of exp
        type(exp[i])
        
        # init setup on exp
        exp[i].setup(
            cut_sub_dfs[i], 
            target = 'rolling_mean_grouped_soil', 
            enforce_exogenous = False, 
            fold_strategy='sliding', 
            fh = fh, 
            session_id = 123, 
            fold = 3,
            ignore_features = ['Time', 'orig_index', 'gradient']
            #numeric_imputation_exogenous = 'mean'
        )
    
    with open('output.txt', 'a') as f:
        # check statistical tests on original data
        for i in range(len(cut_sub_dfs)):
            print("This is the", i, "part of the data:", file=f)
            print(exp[i].check_stats(), file=f)
            
    best = []
    for i in range(len(cut_sub_dfs)):
        print("This is for the", i, "part of the dataset: ")
        best.append(exp[i].compare_models(
            n_select = 5, 
            fold = 3, 
            sort = 'R2',
            verbose = 1, 
            #exclude=['lar_cds_dt','auto_arima','arima'],
            include=['lr_cds_dt', 'br_cds_dt', 'ridge_cds_dt', 
                     'huber_cds_dt', 'knn_cds_dt', 'catboost_cds_dt']
        ))
    
    with open('output.txt', 'a') as f:       
        for i in range(len(best)):
            print("\n The best model, for cut_sub_dfs[", i,"] is:", file=f)
            print(best[i][0], file=f)
    
    return exp, best


# Save the best models
def save_models(exp, best):
    # save pipeline
    model_names = []
    for i in range(len(exp)):
        exp[i].save_model(best[i][0], 'my_first_SPLIT_pipeline_TS_' + str(i))
        model_names.append('my_first_SPLIT_pipeline_TS_' + str(i))
        
    return model_names
        
# TODO: model_names will not work if it was not saved before        
# Load the best models        
def load_models(model_names):
    # load pipeline
    loaded_best_pipeline = []
    for i in range(model_names): # TODO: model_names will not work if it was not saved before
        loaded_best_pipeline.append(load_model(model_names[i]))
    
    return loaded_best_pipeline
    


# Analyze models performance
def analyze_performance(exp, best):
    # plot forecast
    for i in range(len(exp)):
        print("In testset: For the dataset:",i)
        exp[i].plot_model(best[i], plot = 'forecast', save = True)
        #before.save('Plot_in_testset_'+str(i)+'.png', format='png')
        
        print("After testset: For the dataset:",i)
        exp[i].plot_model(best[i], plot = 'forecast', data_kwargs = {'fh' : 500}, save = True)
        #before.save("Plot_after_testset_"+str(i)+".png", format='png')
        
# Tune hyperparameters
def tune_models(exp, best):
    # tune hyperparameters of dt
    tuned_best_models = []
    for i in range(len(best)):
        print("This is for the",i,"model:",best[i])
        tuned_best_models.append(exp[i].tune_model(best[i]))
        
    return best

# Mighty main fuction ;)
def main() -> int:
    # Check version of pycaret, should be >= 3.0
    print("Check version of pycaret:", pycaret.__version__, "should be >= 3.0")
    
    # Data preparation pipeline, calls other subfunction to perform the task
    data, data_plot, data_comb, cut_sub_dfs = prepare_data(csv_filepath)    

    # Start pycaret pipeline: setup, train models, save the best ones to best-array 
    exp, best = create_and_compare_model(cut_sub_dfs)
    
# =============================================================================
#     # Save the best models for further evaluation
#     model_names = save_models(exp, best)
#     
#     # Load model from disk  
#     try:
#         best
#     except NameError:
#         best = load_models(model_names)
# =============================================================================
    
    # Analyze models performance
    #analyze_performance(exp, best)
    
    # Tune hyperparameters
    tuned_best = tune_models(exp, best)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit