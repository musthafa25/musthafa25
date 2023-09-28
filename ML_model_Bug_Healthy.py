import ee
# ee.Authenticate()
ee.Initialize()
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import r2_score
from geemap import ml
import pickle

def read_labeled_data(path1, path2):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df = df1.append(df2)
    df.drop(df.columns[0], axis=1, inplace=True)
    data_list = df.columns
    df = pd.DataFrame(df)
    return df, data_list


def create_train_test(df):
    data_list = df.columns
    data_list = data_list.drop('label')
    data = df
    le = LabelEncoder()
    data['label'] = le.fit_transform(data['label'])
    feature_names = data_list
    label = 'label'
    X = data[feature_names]
    y = data[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.30, random_state=1)
    print('Train', X_train.shape, y_train.shape)
    print('Test', X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test


def rf_modeling(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=10, random_state=15)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("R2:", r2_score(y_test, y_pred))
    return rf


def rf_feature_imp(rf, data_list):
    col_names = data_list.drop('label')
    feature_imp = pd.Series(rf.feature_importances_, index=col_names).sort_values(ascending=False)
    dd_2 = feature_imp
    courses = list(dd_2.keys())
    values = list(dd_2.values)
    with open("features.txt", "w") as output:
        output.write(str(courses))
    return feature_imp, courses, values


def feature_imp_graph(courses, values):
    # creating the bar plot
    fig = plt.figure(figsize=(30, 6))
    plt.bar(courses, values, color='maroon',
            width=0.4)
    plt.xlabel("Features", fontsize=30)
    plt.ylabel("Feature importance", fontsize=30)
    plt.xticks(rotation=90)
    plt.title("Random Forest  Classsifier", fontsize=30)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.show()
    # return features, values

def rf_model_tuning(courses, N):
    df, d_list = read_labeled_data(path1, path2)
    band_list = courses[0:N] # use this for image inference band selection
    data_list = courses[0:N]
    data_list.append('label')
    # data_list
    df_new = df[data_list]
    with open(r'/Users/aidash/home/projects/Bug_infestation/features.txt', 'w') as fp:
        fp.write('\n'.join(band_list))
    return df_new, band_list

def svm_model(X_train, X_test, y_train, y_test, kernel):
    clf = svm.SVC(kernel=kernel)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test) # Predict the response for test dataset
    print('SVM Model Accuracy- ' + kernel)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("R2:", r2_score(y_test, y_pred))
    return clf

def save_ml_models(rf, svm_l, svm_rad):
    out_csv = '/Users/aidash/home/projects/Bug_infestation/data/bug_ml_model.csv'
    trees = ml.rf_to_strings(rf, courses)
    ml.trees_to_csv(trees, out_csv) # This could be converted to GEE Object
    fname1 = '/Users/aidash/home/projects/Bug_infestation/data/bug_ml_model_rf.sav'
    pickle.dump(rf, open(fname1, 'wb'))
    fname2 = '/Users/aidash/home/projects/Bug_infestation/data/bug_ml_model_svm_l.sav'
    pickle.dump(svm_l, open(fname2, 'wb'))
    fname3 = '/Users/aidash/home/projects/Bug_infestation/data/bug_ml_model_svm_rbf.sav'
    pickle.dump(svm_rad, open(fname3, 'wb'))

if __name__ == '__main__':
    path1 = 'data/data_extract/indices_mean_healthy_2020.csv'
    path2 = 'data/data_extract/indices_mean_buginf_2020.csv'
    read_labeled_data(path1, path2)
    # All data sets - Before feature engineering
    df, data_list = read_labeled_data(path1, path2)
    X_train, X_test, y_train, y_test = create_train_test(df)
    rf = rf_modeling(X_train, X_test, y_train, y_test)
    feature_imp, courses, values = rf_feature_imp(rf, data_list)
    # feature_imp_graph(courses, values) # UNCOMMENT IF REQUIRED TO SEE THE FEATURE IMPORTANCE GRAPH

    # Modeling after feature engineering
    df_new, band_list = rf_model_tuning(courses, N=4)
    X_train, X_test, y_train, y_test = create_train_test(df=df_new)
    rf = rf_modeling(X_train, X_test, y_train, y_test)
    print()
    svm_l = svm_model(X_train, X_test, y_train, y_test, kernel='linear')
    print()
    svm_rad = svm_model(X_train, X_test, y_train, y_test, kernel='rbf')
    print(svm_rad)


    # Save RF model as csv to perform inference in gee
    save_ml_models(rf, svm_l, svm_rad)










