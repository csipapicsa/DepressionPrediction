# EATD

https://github.com/speechandlanguageprocessing/ICASSP2022-Depression

folder name:
t - training
v - validation

The SDS (Self-Rating Depression Scale) score is the true label you're looking for. This is a clinical depression screening tool where:

Higher scores indicate more severe depression symptoms
The standardized scores (new_label.txt) are what you should use


You're welcome! Since you have access to the label files, let me help you understand how to interpret them:

The SDS (Zung Self-Rating Depression Scale) scoring typically works as follows:

Raw scores range from 20-80
The standardized scores (in new_label.txt = raw score × 1.25) range from 25-100

For the standardized scores (new_label.txt), the interpretation is usually:

Below 50: Normal range
50-59: Mild depression
60-69: Moderate depression
70 and above: Severe depression


In this dataset, they've used a binary classification:

Depressed
Non-depressed

For your analysis, you should:

Use new_label.txt (the standardized scores) as they're more commonly used in clinical practice
Remember the dataset split:

Training: 83 volunteers (19 depressed, 64 non-depressed)
Validation: 79 volunteers (11 depressed, 68 non-depressed)