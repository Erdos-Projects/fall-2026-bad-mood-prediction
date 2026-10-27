Evaluation Plan \- Bad Mood Prediction

In this project, we use the Lifesnaps dataset, to predict negative moods in individuals based on sleep and exercise data. Our goal is to evaluate how accurately we can predict bad mood in previously unseen individuals using FitBit data. We define this as a classification task: Will the person have a bad mood (1) or non-bad mood (0). A bad mood is defined from three negative mood report factors: sad, tense/anxious and tired, and compared against neutral and positive moods: alert, happy, neutral, and rested/relaxed. We do not differentiate between the three types of bad mood nor do we distinguish between neutral vs good moods. 

We will conduct a participant-level group split using an interactive leave-one-out approach such that on each loop, each participant will be assigned training or test set, not both. This is to not bias the prediction task. This simulates the deployment of our algorithm to new, previously unseen people.

Our primary analysis will focus on hourly mood reports, and will use sleep and exercise measurements to predict these moods. We will also engineer features for the model to account for longer timespans. For instance, cumulative sleep in the past few nights, and cumulative exercise in the last few hours. 

We will report a confusion matrix along with the following metrics: accuracy, balanced accuracy (accounting for class imbalance), precision, recall, and an F1 score summarizing precision and recall. 

|  |  | Actual |  |
| :---- | :---- | :---- | :---- |
|  |  | 1 | 0 |
| Predicted | 1 | Correct Detection | False Alarm |
|  | 0 | Missed Prediction | Correct Rejection |

Fig 1\. An outline of a confusion matrix. As a reminder, our positive class (1) is “bad mood”

We will monitor for outlier sensitivity, distribution differences between the train and test sets, and adversarial leakage. For outlier sensitivity, we will examine whether performance changes when extreme sleep and schedule values are present or excluded according to predefined criteria. For distribution shift, we will compare feature distributions between the training and test sets. Finally, for adversarial leakage, so far our simple majority class predictor achieved 55% accuracy, just slightly better than chance. There is no sign of adversarial leakage so far.