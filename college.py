import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report

training_data = pd.read_csv("CollegeData.csv")

def label_fix(label):
    if label == 'No':
        return 0
    else:
        return 1

training_data['Decision'] = training_data['Decision'].apply(label_fix)

x_data = training_data.drop('Decision', axis = 1)
y_labels = training_data['Decision']

names = tf.feature_column.categorical_column_with_hash_bucket("Names",
                                                        hash_bucket_size = 1000)
embedded_names = tf.feature_column.embedding_column(names, dimension = 1000)

undergrad_num = tf.feature_column.numeric_column("Undergraduates")
first_years_num = tf.feature_column.numeric_column("FirstYears")
admitted = tf.feature_column.numeric_column("PercentAdmitted")
top_percent = tf.feature_column.numeric_column("HSTopPercent")
average_SAT = tf.feature_column.numeric_column("AverageSAT")
median_SAT = tf.feature_column.numeric_column("MiddleSAT")
average_ACT = tf.feature_column.numeric_column("AverageACT")
black_num = tf.feature_column.numeric_column("AfricanAmericans")
asian_num = tf.feature_column.numeric_column("Asians")
white_num = tf.feature_column.numeric_column("Caucasians")
hispanic_num = tf.feature_column.numeric_column("Hispanics")
diversity = tf.feature_column.numeric_column("GeopgraphicDiversity")

feature_cols = [embedded_names, undergrad_num, first_years_num, admitted,
                top_percent, average_SAT, median_SAT, average_ACT, black_num,
                asian_num, white_num, hispanic_num, diversity]

input_func = tf.estimator.inputs.pandas_input_fn(x = x_data, y = y_labels,
                                            batch_size = 10, num_epochs = 1000,
                                            shuffle = True)

dnn_model = tf.estimator.DNNClassifier(hidden_units = [20,20,20],
                                        feature_columns = feature_cols,
                                        n_classes = 2)

dnn_model.train(input_fn = input_func, steps = 1000)

user_input = pd.read_csv("CollegeList.csv")
x_input = user_input.drop('Decision', axis = 1)
y_true = user_input['Decision']

pred_fn = tf.estimator.inputs.pandas_input_fn(x = x_input,
                                              batch_size = len(x_test),
                                              shuffle = False)

predictions = list(model.predict(input_fn = pred_fn))
final_preds = []
for pred in predictions:
    final_preds.append(pred['class_ids'][0])

print(classification_report(y_test, final_preds))    
