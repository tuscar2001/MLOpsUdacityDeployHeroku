

def perform_eda(path):
    try:
        census = pd.read_csv(path + "/census.txt")
        print("Loading the data...")
        col_names = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship",\
                "race","sex","capital-gain","capital-loss","hours-per-week","native-country","probabilities"]
        census.columns = col_names
        print("Import_data: SUCCESS")
        cat_col_corr = ['native-country',"occupation",'workclass']
        for col in cat_col_corr:
            census[col] = census[col].str.strip().replace(r'\s+',' ', regex=True)
            census[col] = census[col].apply(lambda x:"Unknown" if x == "?" else x )
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





def train_models(x_tr, x_ts, y_tr, y_ts):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    print ("Performing a grid search ...")
    rfc = RandomForestClassifier(random_state=42)
    param_grid = { 
    'n_estimators': [100],
    'max_depth' : [3]}

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_tr, y_tr)
    x_data = pd.concat([x_tr, x_ts], ignore_index = True)
    # feature_importance_plot(cv_rfc, x_data, './3 model/output/')
    # print ("Plotting ROC curves ...")
    print ("RF best model {}".format(cv_rfc.best_estimator_))
    # plot_model(lrc, cv_rfc, x_ts, y_ts)
    y_tr_preds_rf = cv_rfc.best_estimator_.predict(x_tr)
    y_ts_preds_rf = cv_rfc.best_estimator_.predict(x_ts)
    print ("Saving the best estimator for Random Forest ...")
    joblib.dump(cv_rfc.best_estimator_, '/home/tuscar2022/Documents/Udacity MLOps projects/deployMLHeroku/deployHeroku/3 model/output/model.pkl')
    return y_tr_preds_rf, y_ts_preds_rf


def classification_report_txt(report):
    lines = list(report.split('\n'))
    if len(lines) == 9:
        title = [i for i in lines[0].split(" ") if i !=""]
        title.insert(0, 'class')
        row1 = [i for i in lines[2].split(" ") if i != ""]
        row2 = [i for i in lines[3].split(" ") if i != ""]
        row3 = [i for i in lines[5].split(" ") if i != ""]
        row = f"Slice:{j} ,Class:{row1[0]} ,Precision:{row1[1]} ,Recall:{row1[2]} ,f1-score:{row1[3]} ,Support:{row1[4]} ---> Class:{row2[0]} ,Precision:{row2[1]} ,Recall:{row2[2]} ,f1-score:{row2[3]} ,Support:{row2[4]} Accuracy:{row3[1]}"
        evaluate_slices.append(row)

        with open("/home/tuscar2022/Documents/Udacity MLOps projects/deployMLHeroku/deployHeroku/3 model/output/results.txt", "w") as result:
            for line in evaluate_slices:
                result.write(line + '\n')




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
    from sklearn.metrics import plot_roc_curve, classification_report
    from dmba.metric import AIC_score
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    import logging

    logging.basicConfig(
    filename='/home/tuscar2022/Documents/Udacity MLOps projects/deployMLHeroku/deployHeroku/logs/train.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

    path = "/home/tuscar2022/Documents/Udacity MLOps projects/deployMLHeroku/deployHeroku/1 data/"

    in_data = perform_eda(path)
    in_data = import_processed_data(path)
    cat_col =  [col for col in in_data.columns if in_data[col].dtype=="O"]
    num_col = [col for col in in_data.columns if in_data[col].dtype!="O"]
    X, y = encoder_helper(in_data, cat_col, num_col)
    x_train, x_test, y_train, y_test = perform_feature_engineering(X, y)
    y_train_preds_rf, y_test_preds_rf = train_models(x_train,x_test,y_train,y_test)

    model = joblib.load('/home/tuscar2022/Documents/Udacity MLOps projects/deployMLHeroku/deployHeroku/3 model/output/model.pkl')

    evaluate_slices = []
    for j in in_data.education.unique():
        sliced = in_data[in_data.education == j]
        X, y = encoder_helper(sliced, cat_col, num_col)
        x_train, x_test, y_train, y_test = perform_feature_engineering(X, y)
        y_train_preds_rf, y_test_preds_rf = train_models(x_train,x_test,y_train,y_test)
        report = classification_report(y_test,y_test_preds_rf,zero_division=0)
        classification_report_txt(report)
            
