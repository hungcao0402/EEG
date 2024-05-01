# Project Name
> Spelling Errors Detection by Electroencephalography (EEG)

## Introduction
When humans think, we produce brain waves, which can be mapped to actual intentions. This project, we use brain waves data of people with the goal of spelling a word by only paying attention to visual stimuli.

The goal of this project is to detect errors during the spelling task, given the subject's brain waves.

## Data
- ChannelsLocation.csv: information for each channel to a topographical representation of multichannel EEG.
- train.zip: Training set made of 16 subjects who had gone through 5 sessions, for a total of 80 Data_S*_Sess*.csv files. 60 feedbacks were provided in each session except the fifth one for which 100 feedbacks were provided.
- TrainLabels.csv: the expected labels  for the training set.
    - IdFeedBack: Identifier for each feedback (e.g., S01_Sess01_FB001 corresponds to the first feedback in the session 01 for the subject 01)
    - Prediction: 0 or 1 for bad or good feedback, respectively.
- test.zip: Test set made of 10 other subjects who had gone through 5 sessions, for a total of 50 Data_S*_Sess*.csv files
- SampleSubmission.csv: Sample submission file in the correct format

(Details on https://www.kaggle.com/competitions/inria-bci-challenge/data)

## Method
We propose two different models. The first one does not make any use of the leakage information and satisfies an "online processing" constraint, which means that any trial performed by a subject can be classified without the need for future complementary data or information. The second model uses the leak, it added 2 other features ("OnlineErr" and "Err Prop" in metadata file), thus it is not online-compatible. The two models are built upon the same classification pipeline, but with parameters tuned independently to achieve the highest performance.

## Usage:
1. Prepare data:
Download data in https://www.kaggle.com/competitions/inria-bci-challenge/data.
Extract train.zip to "train" and test.zip to "test" in folder "data". Put file TrainLabels.csv in the train folder.

2. Create conda environment with required packages:
```bash
conda env create -f eeg.yml
conda activate eeg
```

3. Run preproc.py
```bash
cd preproc/
python preproc.py
```
4. Generating Submissions:
For the submission with the leak:
```bash
python prediction.py parameters_Leak.yaml
```
For the submission without the leak:
```bash
python prediction.py parameters_noLeak.yaml
```

## Evaluate results:
Result is a submission .csv file. You can push it into above kaggle link to check AUC score for private test.

Other results will be made if you change some parameters in parameters_Leak.yaml or parameters_noLeak.yaml file.