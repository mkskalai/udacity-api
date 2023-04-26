# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model was developed by Maksym Kalaidov on 19th of April, 2023, as a part of the final project for Udacity's MLOps Deployment course.
This is a binary classifier, based on sklear's Random Forest Classifier. For more info check out: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
The only hyperparameter changed from default is max_depth (set to 10), but with some tuning a much better model can be built.
Trained model can be found under starter/model/random_forest.joblib

## Intended Use
This model can be used to classify a person's income into "less than 50k" or "more than 50k" based on their background.
It only makes sense to use for this the data it was trained on: https://archive.ics.uci.edu/ml/datasets/census+income, since countries other than US would have different data, and, this data is not up-to-date.

## Training Data
For this project, Census income data was used. Please find more details about the data here: https://archive.ics.uci.edu/ml/datasets/census+income.
The subset actually used for training can be found in the project folder, under starter/data/data_train.csv

## Evaluation Data
Evaluation data can be found in the project folder, under starter/data/data_test.csv

## Metrics
3 metrics were used to evaluate this model: presision, recal and fbeta score.

Overall performance:
precision: 0.816     recall: 0.540     fbeta: 0.650 

In addition, all metrics were computed for each unique value fixed for each cathegorical variable of the data. Results can be found in the project folder under starter/model/slice_output.txt

## Ethical Considerations
After checking the model performance for different cathegories of people, it is clear that the model performs very different for different groups. It is possible, that the model is highly biased, and should be used with caution.

## Caveats and Recommendations
With some time spent on hyperparameter tuning, a much better model can bu built. In addition, building a reproducible ML pipeline would allow for easier experimentation.
