# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model uses GradientBoostingClassifier from sklearn.ensemble for a classification task. The parameters are the best from a grid search.

## Intended Use

Given a set of census individual characteristics, the model predicts whether a person makes over 50K USD a year.

## Training Data

The data is downloaded from https://archive.ics.uci.edu/dataset/20/census+income, where more information can be found. The features are the typical individual charactertics, including, for example, age, education and matrital status.

## Evaluation Data

The original data is first processed and then split into training and evaluation data with a test_size of 0.2.

## Metrics
_Please include the metrics used and your model's performance on those metrics._

precision: 0.7882986913010007 
recall: 0.649746192893401
fbeta: 0.7123478260869565

## Ethical Considerations

Whether the model is baised depends on whether the original data is biased. Given that it is census data, the possibility of biasing against any particular group is low.

## Caveats and Recommendations

The model predicts whether a person makes over 50K USD a year. However, this number is nominal and does not adjust with inflation. A more meaningful task is to predict whether a person makes over some amount of real income a year.
