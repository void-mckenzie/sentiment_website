# Sentiment Analysis of Movie Reviews

For this project 50,000 IMDB Reviews were used to build a RNN-LSTM model for analysis the sentiment of movie reviews.

The README_IMP files in the folders provide the links required for downloading the IMDB dataset and the GLoVE embedding.

Additionally, the pre-trained model weights can be downloaded from the link provided in the readME file, present in the saved_models folder.

The Preprocessing_Training code performs the various preprocessings (parsing through dataset, remove punctuations, stop words, tokenization, glove embedding etc.) required to format the reviews. It also defines the model architecture and trains the model.

The deployment code is to be run on the flask server. It loads the trained model and serves it. After the server has been started, the "home_page.html" file can be used to access the website, where you can type in a review and get the appropriate prediction.
<p>
  <img src="Doc/demo.png">
  </p>

The model has an accuracy of 92.5%. We are working on using the TF2.0 serving API to serve the model directly as an API.
