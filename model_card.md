# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model uses GradientBoostingClassifier from sklearn.ensemble for a classification task. The parameters are the best from a grid search.

## Intended Use

Given a set of census individual characteristics, the model predicts whether a person makes over 50K USD a year.

## Training Data

The data is from https://archive.ics.uci.edu/dataset/20/census+income, where more information can be found. The features are typical individual characteristics, for example, age, education and marital status.

## Evaluation Data

The original data is cleaned and then split into training and testing data with a test_size of 0.2.

## Metrics

precision: 0.79

recall: 0.65

fbeta: 0.71

## Ethical Considerations

Whether the model is biased depends on whether the original data is biased. Given that it is census data, the possibility of biasing against any particular group is low.

## Caveats and Recommendations

The model predicts whether a person makes over 50K USD a year. However, this number is nominal and is not inflation-adjusted. A more meaningful task is to predict whether a person makes some amount of real income a year, or his position in the income distribution.
