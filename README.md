# CSE559DataSelectionProject


This project focuses on the performance of the embedding model catELMo and if utilizing data selection strategies would increase the performance of the model on the downstream task of predicting whether the epitope and TCR pair is a bonding or non-bonding pair. The data selection strategies used in this project are the following:
- random selection
- k-means clustering selection
- length variation
- frequency weighted k-mer selection
- distance-based diversity

All of these methods are used to select 60% of the initial data folder 'train_prefix.' Due to computational time, the downstream task is then done on 7% of the given TCR-Epitope pairs. The prediction model is trained for a total of 10 times, 5 per split (epitope and TCR).

The results of our testing and training is found here: [Link](https://docs.google.com/spreadsheets/d/1mLklM-94pne8dZLWk7399l-acLAba-86AILmu1bicnc/edit?usp=sharing)
