import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

print("Tensorflow v" + tf.__version__)
print("Tensorflow Decision Forests v" + tfdf.__version__)
dataset_df = pd.read_csv('../data/train.csv')
print(dataset_df.head(5))

print("0" * 100)
print(dataset_df.columns)
print("2" * 100)
# There are 12 feature columns. Using these features your model has to predict whether the passenger is rescued or not indicated by the column Transported.
#
# add Codeadd Markdown
# Let us quickly do a basic exploration of the dataset
print(dataset_df.describe())
print(dataset_df.info())
# Bar chart for label column: Transported¶
plot_df = dataset_df.Transported.value_counts()
plot_df.plot(kind='bar')
# Numerical data distribution¶
# Let us plot all the numerical columns and their value counts:

fig, ax = plt.subplots(5, 1, figsize=(10, 10))
plt.subplots_adjust(top=10)
sns.histplot(dataset_df['Age'], bins=50, color='b', ax=ax[0])
sns.histplot(dataset_df['FoodCourt'], bins=50, color='b', ax=ax[1])
sns.histplot(dataset_df['ShoppingMall'], bins=50, color='b', ax=ax[2])
sns.histplot(dataset_df['Spa'], bins=50, color='b', ax=ax[3])
sns.histplot(dataset_df['VRDeck'], bins=50, color='b', ax=ax[4])
# Prepare the dataset¶
# We will drop both PassengerId and Name columns as they are not necessary for model training.

dataset_df = dataset_df.drop(['PassengerId', 'Name'], axis=1)
print("4" * 100)
print(dataset_df.head(5))

# plt.show()
# We will check for the missing values using the following code:
dataset_df.isnull().sum().sort_values(ascending=False)
# This dataset contains a mix of numeric, categorical and missing features. TF-DF supports all these feature types natively, and no preprocessing is required.
#
# But this datatset also has boolean fields with missing values. TF-DF doesn't support boolean fields yet. So we need to convert those fields into int. To account for the missing values in the boolean fields, we will replace them with zero.
#
# In this notebook, we will replace null value entries with zero for numerical columns as well and only let TF-DF handle the missing values in categorical columns.
#
# Note: You can choose to let TF-DF handle missing values in numerical columns if need be.
dataset_df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = dataset_df[[
    'VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'
]].fillna(value=0)

label = 'Transported'
dataset_df[label] = dataset_df[label].astype(int)
# We will also convert the boolean fields CryoSleep and VIP to int.
print("5" * 100)
dataset_df['VIP'] = dataset_df['VIP'].astype(int)
dataset_df['CryoSleep'] = dataset_df['CryoSleep'].astype(int)
# The value of column Cabin is a string with the format Deck/Cabin_num/Side. Here we will split the Cabin column and create 3 new columns Deck, Cabin_num and Side, since it will be easier to train the model on those individual data.
# Run the following command to split the column Cabin into columns Deck, Cabin_num and Side
dataset_df[["Deck", "Cabin_num", "Side"]] = dataset_df['Cabin'].str.split('/', expand=True)
# Remove original Cabin column from the dataset since it's not needed anymore.
print("6" * 100)
print(dataset_df.columns)
try:
    dataset_df.drop('Cabin', inplace=True, axis=1)
except KeyError:
    print('Field does not exists!')

dataset_df.head(5)


def split_dataset(dataset, test_ratio=0.20):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]


train_ds_pd, valid_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples in testing.".format(
    len(train_ds_pd), len(valid_ds_pd)))

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label)
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label=label)
'''
Select a Model¶
There are several tree-based models for you to choose from.

RandomForestModel
GradientBoostedTreesModel
CartModel
DistributedGradientBoostedTreesModel
To start, we'll work with a Random Forest. This is the most well-known of the Decision Forest training algorithms.

A Random Forest is a collection of decision trees, each trained independently on a random subset of the training dataset (sampled with replacement). The algorithm is unique in that it is robust to overfitting, and easy to use.

We can list the all the available models in TensorFlow Decision Forests using the following code:'''
print(tfdf.keras.get_all_models())
'''
Configure the model
TensorFlow Decision Forests provides good defaults for you (e.g. the top ranking hyperparameters on our benchmarks, slightly modified to run in reasonable time). If you would like to configure the learning algorithm, you will find many options you can explore to get the highest possible accuracy.

You can select a template and/or set parameters as follows:

rf = tfdf.keras.RandomForestModel(hyperparameter_template="benchmark_rank1")

Read more here.'''
# rf = tfdf.keras.RandomForestModel(hyperparameter_template='benchmark_rank1')
# Create a Random Forest¶
# Today, we will use the defaults to create the Random Forest Model. By default the model is set to train for a classification task.
# Train the model¶
# We will train the model using a one-liner.
#
# Note: you may see a warning about Autograph. You can safely ignore this, it will be fixed in the next release.

rf = tfdf.keras.RandomForestModel()
rf.compile(metrics=["accuracy"])  # Optional, you can use this to include a list of eval metrics
rf.fit(x=train_ds)

# Visualize the model¶
# One benefit of tree-based models is that we can easily visualize them. The default number of trees used in the Random Forests is 300. We can select a tree to display below.
# tfdf.model_plotter.plot_model_in_colab(rf, tree_idx=0, max_depth=3)
'''
Evaluate the model on the Out of bag (OOB) data and the validation dataset¶
Before training the dataset we have manually seperated 20% of the dataset for validation named as valid_ds.

We can also use Out of bag (OOB) score to validate our RandomForestModel. 
To train a Random Forest Model, a set of random samples from training set are choosen by the algorithm 
and the rest of the samples are used to finetune the model.
The subset of data that is not chosen is known as Out of bag data (OOB). 
OOB score is computed on the OOB data.

Read more about OOB data here.

The training logs show the accuracy evaluated on the out-of-bag dataset according to the number of trees in the model.
 Let us plot this.

Note: Larger values are better for this hyperparameter.
'''
import matplotlib.pyplot as plt

logs = rf.make_inspector().training_logs()
plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
plt.xlabel('Number of trees')
plt.ylabel('Accuracy ( out-of-bag)')
plt.show()
print("1" * 100)
inspector = rf.make_inspector()
inspector.evaluation()
evaluation = rf.evaluate(x=valid_ds, return_dict=True)
for name, value in evaluation.items():
    print(f"{name}:{value:.4f}")
# Variable importances¶
# Variable importances generally indicate how much a feature contributes to the model predictions or quality. There are several ways to identify important features using TensorFlow Decision Forests. Let us list the available Variable Importances for Decision Trees:
print(f"Available variable importance")
for importance in inspector.variable_importances().keys():
    print('\t', importance)

# As an example, let us display the important features for the Variable Importance NUM_AS_ROOT.
# The larger the importance score for NUM_AS_ROOT, the more impact it has on the outcome of the model.
# By default, the list is sorted from the most important to the least. From the output you can infer that the feature at the top of the list is used as the root node in most number of trees in the random forest than any other feature.
print("5" * 100)
print(inspector.variable_importances()['NUM_AS_ROOT'])

# Submission
test_df = pd.read_csv('../data/test.csv')
submission_id = test_df.PassengerId
test_df[['VIP', 'CryoSleep']] = test_df[['VIP', 'CryoSleep']].fillna(value=0)
test_df[["Deck", "Cabin_num", "Side"]] = test_df['Cabin'].str.split('/', expand=True)
test_df = test_df.drop('Cabin', axis=1)
test_df['VIP'] = test_df['VIP'].astype(int)
test_df['CryoSleep'] = test_df['CryoSleep'].astype(int)

test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df)
n_predictions = rf.predict(test_ds)
output = pd.DataFrame({'PassengerId': submission_id,
                       'Transported': n_predictions.squeeze()})
print("6" * 100)
print(output.head())

sample_submission_df = pd.read_csv('../data/sample_submission.csv')
sample_submission_df['Transported'] = n_predictions
sample_submission_df.to_csv('../data/sample_submission_01.csv', index=False)
print("7" * 100)
print(sample_submission_df.head())
