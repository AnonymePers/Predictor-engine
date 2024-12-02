import os
import pickle
import sys
import graphviz
from xgboost import XGBClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn import tree
import numpy as np
from nonconformist.icp import IcpClassifier
from nonconformist.nc import NcFactory
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier
from nonconformist import icp
from pm4py.objects.conversion.log import converter as log_converter
import settings
from src.dataset_manager import DatasetManager
from src.machine_learning import get_encoder, freq_encode_traces
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_svm(X_train, y_train):
    param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__kernel': ['linear', 'rbf'],
        'svc__gamma': ['scale', 'auto']
    }

    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Feature scaling
        ('svc', SVC())  # SVM classifier
    ])

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

def train_decision_tree(X_train, y_train):
    param_grid = {
        'max_depth': [4, 6, 8, 10, None],
        'class_weight': [None, 'balanced'],
        'min_samples_split': [0.1, 0.2, 0.3, 2],
        'min_samples_leaf': [1, 10, 16]
    }

    dt = DecisionTreeClassifier()
    grid_search = GridSearchCV(dt, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

def train_lgbm(X_train, y_train):
    param_grid = {
        'num_leaves': [31, 63, 127],
        'max_depth': [-1, 5, 10, 15],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'class_weight': [None, 'balanced']
    }

    lgbm = LGBMClassifier()
    grid_search = GridSearchCV(lgbm, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_


def train_xgboost(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 1, 5],
        'scale_pos_weight': [1, 10, 25]  # Useful for imbalanced datasets
    }

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')  # Disable deprecation warnings
    grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{model_name} Results:")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("\n")

def run_experiment(log_data):
    print("start encoding")
    encoder_args = {'case_id_col': dataset_manager.case_id_col,
                    'static_cat_cols': dataset_manager.static_cat_cols,
                    'static_num_cols': dataset_manager.static_num_cols,
                    'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                    'dynamic_activity_col': dataset_manager.dynamic_activity_col,
                    'dynamic_num_cols': dataset_manager.dynamic_num_cols,
                    'label_col': dataset_manager.label_col,
                    'pos_label': dataset_manager.pos_label,
                    'fillna': True}
    encoder = get_encoder(method, settings.support_threshold_dict, **encoder_args)
    X, y = encoder.fit_transform(log_data)

    '''data = log_data.rename(columns={dataset_manager.timestamp_col: 'time:timestamp',
                                dataset_manager.case_id_col: 'case:concept:name',
                                dataset_manager.activity_col: 'concept:name'})
    data = log_converter.apply(data)
    features, encoded_data = freq_encode_traces(data)
    X2 = pd.DataFrame(encoded_data, columns=features)
    X = pd.concat([X1, X2], axis=1)'''

    # split into training and test

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2)
    X_train_val = X_train_val.drop("prefix", axis=1)
    X_train, X_cal, y_train, y_cal = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
    X_test = X_test.drop("prefix", axis=1)

    print("start training")
    '''    print("train DT")
    dt_model = train_decision_tree(X_train, y_train)
    with open(os.path.join(settings.output_dir, f'{dataset_name}_DT.pickle'),
              'wb') as file:
        pickle.dump((dt_model), file)
    # Train models
    print("train Xgboost")
    xgboost_model = train_xgboost(X_train, y_train)
    with open(os.path.join(settings.output_dir, f'{dataset_name}_xgboost.pickle'),
              'wb') as file:
        pickle.dump((xgboost_model), file)'''

    svm_model = train_svm(X_train, y_train)
    with open(os.path.join(settings.output_dir, f'{dataset_name}_svm.pickle'),
              'wb') as file:
        pickle.dump((svm_model), file)

    try:
        with open(os.path.join(settings.output_dir, f'{dataset_name}_DT.pickle'), 'rb') as file:
            dt_model = pickle.load(file)
            
    except FileNotFoundError as not_found:
        print(f"Model {dataset_name}.pickle not found. Invalid path or you "
              f"have to train a model before loading.")
        sys.exit(2)
    try:
        with open(os.path.join(settings.output_dir, f'{dataset_name}_xgboost.pickle'), 'rb') as file:
            xgboost_model = pickle.load(file)
    except FileNotFoundError as not_found:
        print(f"Model {dataset_name}.pickle not found. Invalid path or you "
              f"have to train a model before loading.")
        sys.exit(2)
    dot_data = tree.export_graphviz(
        dt_model,
        out_file=None,
        feature_names=X_train.columns,
        class_names=['Adapt', 'noAdapt'],
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = graphviz.Source(dot_data, format="pdf")
    graph.render(os.path.join(settings.output_dir, f'DT_{dataset_name}'))
    print("start evaluating")
    # Evaluate models
    evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    evaluate_model(svm_model, X_test, y_test, "SVM")
    evaluate_model(xgboost_model, X_test, y_test, "XGBoost")

    # Setup conformal prediction for best performing model
    best_model = select_best_model([dt_model,svm_model, xgboost_model ], X_test, y_test)
    best_model=dt_model
    # Create a nonconformity measure
    nc = NcFactory.create_nc(best_model)

    # Wrap the classifier in an IcpClassifier
    icp = IcpClassifier(nc)

    # Fit the IcpClassifier
    icp.fit(X_train, y_train)
    icp.calibrate(X_cal, y_cal)
    rf_model = icp.nc_function.model.model

    # Removing cal data from the model
    icp.cal_x = []
    icp.cal_y = []

    test_pval = icp.predict(X_test.values)
    # print(test_pval)
    sys.path.append("plot_utils/python/src/")
    from plot_utils.python.src.pharmbio.cp import metrics
    from plot_utils.python.src.pharmbio.cp import plotting

    test_pval = pd.DataFrame(test_pval, columns=['p_value_0', 'p_value_1'])

    results_df = pd.DataFrame({
        'true_labels': y_test.values,
        'p_value_0': test_pval['p_value_0'],
        'p_value_1': test_pval['p_value_1']
    })

    results_df.to_csv(settings.output_dir + '/C_DT_predictions.csv', sep=';', index=False)

    print(f"Done with: {dataset_name}...\n")


def select_best_model(models, X_test, y_test):
    best_accuracy = 0
    best_model = None

    for model in models:
        accuracy = accuracy_score(y_test, model.predict(X_test))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    return best_model


if __name__ == "__main__":
    # Load your data
    dataset_name = "FMPlog"
    dataset_manager = DatasetManager(dataset_name.lower())
    data = dataset_manager.read_dataset(os.path.join(os.getcwd(), settings.dataset_folder))
    constr_family = "latestindex"
    method = "latestindex"

    # determine min and max (truncated) prefix lengths
    min_prefix_length = 1
    max_prefix_data = dataset_manager.get_pos_case_length_quantile(data, 0.90)

    data = dataset_manager.generate_prefix_data(data=data, min_length=1, max_length=max_prefix_data, gap=1)


    # Run the experiment
    run_experiment(data)