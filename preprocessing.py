import pandas as pd
import numpy as np
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def load_datasets(dataset_name='bank'):

    if dataset_name == 'bank':
        dataset = 'bank_data.csv'
        label_col = 'y'
        delim = ';'
        label_map = {'no': 0, 'yes': 1}
    elif dataset_name == 'maternal':
        dataset = 'maternal_data.csv'
        label_col = 'RiskLevel'
        delim = ','
        label_map = {'low risk': 0, 'mid risk': 1, 'high risk': 2}
    elif dataset_name == 'winequality':
        dataset = 'winequality_data.csv'
        label_col = 'quality'
        delim = ';'
        label_map = dict([(i, i - 3) for i in range(3, 9)])

    curr_data = pd.read_csv('datasets/{0}'.format(dataset), sep=delim)

    # Convert string targets into actual unique ints and same target col name
    curr_data[label_col] = curr_data[label_col].apply(lambda x: label_map[x])
    curr_data = curr_data.rename({label_col: 'y'}, axis='columns')

    return curr_data

def read_and_split_data(dataset_name='bank'):
    curr_data = load_datasets(dataset_name=dataset_name)
    if dataset_name == 'bank':
        print("One hot encoding categorical features...")
        cat_cols = curr_data.select_dtypes(["object"]).columns
        # Can use this if sklearn causes need to change other code, but usually worse for inference
        # curr_data = pd.get_dummies(curr_data, columns=cat_cols)

        transform = ColumnTransformer(
                transformers=[('onehotenc',OneHotEncoder(),cat_cols)],
                remainder='passthrough',
                verbose_feature_names_out = False)

        curr_data = pd.DataFrame(
                transform.fit_transform(curr_data),
                columns = transform.get_feature_names_out())

        # Just so the reporting does not have to change :)
        curr_data['y'] = curr_data['y'].astype(int)
        
    x_data = np.array(curr_data.loc[:, curr_data.columns != 'y'])
    y_data = np.array(curr_data['y']).flatten()

    # [train, val, test] = [0.6, 0.2, 0.2]; can change later
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=573, shuffle=False)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=573, shuffle=False)
    return x_train, y_train, x_val, y_val, x_test, y_test

def target_dist(dataset):

    target_counts = dataset["y"].value_counts().sort_index().values
    target_ratio = target_counts / np.sum(target_counts)
    print(target_counts)
    print(target_ratio)

def measure_complexity(dataset):
    # Need to be more careful for bank data -- one hot encoding doesnt work as well with PCA, probably should ordinal ecnode + circular encode
    X = dataset.drop(['y'], axis=1)
    pca = decomposition.PCA(n_components=None)
    pca.fit(X)
    explained_var = pca.explained_variance_ratio_
    explained_var_cummulative = np.cumsum(explained_var)

    print(sum(explained_var_cummulative < 0.95)/len(explained_var_cummulative))
    # Count components needed to explain 95% of variance
    #print(explained_var_cummulative)

if __name__ == '__main__':
    for dataset_name in ['bank', 'maternal', 'winequality']:
        dataset = load_datasets(dataset_name=dataset_name)

        # Measure imbalance
        target_dist(dataset)

        if dataset_name != 'bank':
            measure_complexity(dataset)
