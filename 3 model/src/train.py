

def perform_eda(path):
    try:
        census = pd.read_csv(path + "/census.txt")
        print("Loading the data...")
        col_names = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship",\
                "race","sex","capital-gain","capital-loss","hours-per-week","native-country","probabilities"]
        census.columns = col_names
        print("Import_data: SUCCESS")
        census.to_csv(path + '/cleaned_census.csv', index = False)
    except FileNotFoundError as err:
            print ("Error: File not find in the given path ...")
    
    

def import_processed_data(path):
    '''
    returns dataframe for the csv found in the path

    input:
        path: path to the census data
    output:
        in_data: pandas dataframe
    '''
    try:
        in_data = pd.read_csv(path + '/cleaned_census.csv')
        print ("SUCCESS: File read correctly ...")
        return in_data
    except FileNotFoundError:
        print ("Error: File not found in the given path ...")



    


def encoder_helper(in_data_hp, cat_lst, num_list):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook
    input:
            in_data: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could
             be used for naming variables or index y column]
    output:
            in_data: pandas dataframe with new columns for
    '''
    print ("Dummy categorical freatures ...")
    cat_lst = [i for i in cat_lst if i != 'probabilities']
    cat_df = pd.get_dummies(in_data_hp[cat_lst])
    num_df = in_data_hp[num_list]
    X = pd.concat([cat_df,num_df],axis = 1)
    y = in_data_hp['probabilities']
    return X, y


def perform_feature_engineering(x_dsn,y_dsn):
    '''
    input:
              in_data: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]
    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    print ('Splitting input data into training and testing ...')
    x_tr, x_ts, y_tr, y_ts = train_test_split(x_dsn,y_dsn,test_size= 0.3,random_state=42)
    print (x_tr.shape,y_tr.shape,x_ts.shape,y_ts.shape)
    return x_tr,x_ts,y_tr,y_ts  



if __name__ == "__main__":
    import joblib
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    import matplotlib.pylab as plt
    from dmba import classificationSummary, gainsChart, liftChart
    from dmba.metric import AIC_score

    path = "/home/tuscar2022/Documents/Udacity MLOps projects/deployMLHeroku/deployHeroku/1 data/"

    in_data = perform_eda(path)
    in_data = import_processed_data(path)
    cat_col =  [col for col in in_data.columns if in_data[col].dtype=="O"]
    num_col = [col for col in in_data.columns if in_data[col].dtype!="O"]
    X, y = encoder_helper(in_data, cat_col, num_col)
    x_train, x_test, y_train, y_test = perform_feature_engineering(X, y)