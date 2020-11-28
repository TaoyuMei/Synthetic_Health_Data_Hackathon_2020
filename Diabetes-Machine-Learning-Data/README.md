# Diabetes - Machine Learning - Data Set

This challenge focuses on building predictive machine learning models with synthetic data. The focus here is not so much on the biology behind the data, but on the synthetic data’s abilities/limitations in the context of machine learning. 

Possible paths to follow:
* Train a model on synthetic data to predict hospital readmission (or other interesting features), and see if predictions are accurate on real data.
* Train two models: one on synthetic and one on real data. Compare predictions and see if any information is lost with synthetic data. 
* Train a classifier that can differentiate between real and synthetic data. 

**Aim:** Investigate how well features, such as hospital re-admission, can be predicted using synthetic data.

## Data
##### Note: To access the data set, Download or Fork this project (on the left under Source Files). Due to the file size (38MB), it may take a few minutes to download.

You are provided two data sets: 1 set of real patient data (`real_data.csv`) and 1 set of synthetic patient data (`synthetic_data.csv`). 
Both data sets consists of 78441 rows with 42 columns. 
The files can be found in the `data` dictionary in the source files.

The feature descriptions can be found in `data/feature_descriptions.csv`: 

- `race` - Values: Caucasian, Asian, African American, Hispanic, and other
- `gender` - Values: male and female
- `age` - Grouped in 10-year intervals: [0, 10), [10, 20), . . ., [90, 100)
- `time_in_hospital` - Integer number of days between admission and discharge
- `num_lab_procedures` - Number of lab tests performed during the encounter
- `num_procedures` - Number of procedures (other than lab tests) performed during the encounter
- `num_medications` - Number of distinct generic names administered during the encounter
- `number_outpatient` - Number of outpatient visits of the patient in the year preceding the encounter
- `number_emergency` - Number of emergency visits of the patient in the year preceding the encounter
- `number_inpatient` - Number of inpatient visits of the patient in the year preceding the encounter
- `number_diagnoses` - Number of diagnoses entered to the system
- `max_glu_serum` - Indicates the range of the result or if the test was not taken. Values: >200, >300, normal, and none if not measured
- `A1Cresult` - Indicates the range of the result or if the test was not taken. Values: >8 if the result was greater than 8%, >7 if the result was greater than 7% but less than 8%, normal if the result was less than 7%, and none if not measured
- `metformin` - Medication
- `repaglinide` - Medication
- `nateglinide` - Medication
- `chlorpropamide` - Medication
- `glimepiride` - Medication
- `acetohexamide` - Medication
- `glipizide` - Medication
- `glyburide` - Medication
- `tolbutamide` - Medication
- `pioglitazone` - Medication
- `rosiglitazone` - Medication
- `acarbose` - Medication
- `miglitol` - Medication
- `troglitazone` - Medication
- `tolazamide` - Medication
- `examide` - Medication
- `citoglipton` - Medication
- `insulin` - Medication
- `glyburide-metformin` - Medication
- `glipizide-metformin` - Medication
- `glimepiride-pioglitazone` - Medication
- `metformin-rosiglitazone` - Medication
- `metformin-pioglitazone` - Medication
- `change` - Indicates if there was a change in diabetic medications (either dosage or generic name). Values: change and no change
- `diabetesMed` - Indicates if there was any diabetic medication prescribed. Values: yes and no
- `readmitted` - 30 days, >30 if the patient was readmitted in more than 30 days, and No for no record of readmission
- `_diag_1` - Generic primary diagnosis extracted from ICD9 codes
- `_diag_2` - Generic secondary diagnosis extracted from ICD9 codes
- `_diag_3` - Additional generic secondary diagnosis extracted from ICD9 codes

## FAQ
Please join the #diabetes-machine-learning room on Slack.
