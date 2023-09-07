# rnn_raps

This is a project that implements an LSTM to generate rap lyrics. 

Data source is rap lyrics from Kaggle [dataset](https://www.kaggle.com/code/rikdifos/rap-lyrics-text-mining). 
Text is preprocessed via tokenizing and splitting into RNN-style time-series data splits. 
The model consists of a custom embedding layer based off of GloVe 100d, LSTM, and a dense layer.

Preprocesing and model training is performed on [Great Lakes](https://arc.umich.edu/greatlakes/user-guide/) High Performance Computing GPU clusters.
