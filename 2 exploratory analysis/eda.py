import pandas as pd








try:
    census = pd.read_csv("/home/tuscar2022/Documents/Udacity MLOps projects/deployMLHeroku/deployHeroku/1 data/census.txt")
    print("Loading the data...")
    col_names = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship",\
             "race","sex","capital-gain","capital-loss","hours-per-week","native-country","probabilities"]
    census.columns = col_names
    logging.info("Import_data: SUCCESS")
except FileNotFoundError as err:
        print ("Error: File not find in the given path ...")
        logging.error("Issue found while loading the data ...")


cat_col_corr = ['native-country',"occupation",'workclass']
for col in cat_col_corr:
    census[col] = census[col].str.strip().replace(r'\s+',' ', regex=True)
    census[col] = census[col].apply(lambda x:"Unknown" if x == "?" else x )


census.to_csv('/home/tuscar2022/Documents/Udacity MLOps projects/deployMLHeroku/deployHeroku/1 data/cleaned_census.csv')




