# Sentiment Analysis App

This is a web application built using Streamlit that performs sentiment analysis on tweets. The app classifies the emotional tone of the input text as positive or negative, providing valuable insights into customer opinions.

## Features

- **User-friendly Interface**: Simple input field for entering tweets.
- **Sentiment Classification**: Classifies the sentiment of the input text as positive or negative.
- **Real-time Predictions**: Get instant feedback on sentiment analysis.

## Technologies Used

- **Streamlit**: For building the web application.
- **TensorFlow/Keras**: For the machine learning model.
- **NLTK**: For natural language processing tasks like tokenization, stemming, and stopword removal.
- **Pickle**: For loading pre-trained models and tokenizers.

## Installation

To run this application locally, you need to have Python installed. Follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-app.git
   cd sentiment-analysis-app
   ```

2. Install the required libraries:
   ```bash
   pip install streamlit tensorflow nltk
   ```

3. Download necessary NLTK resources:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

4. Place your `model.pkl` and `token.pkl` files in the same directory as the app.

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Open the application in your web browser.
2. Enter the tweet you want to analyze in the input field.
3. Click the "Predict" button to see the sentiment classification.

## Model Details

The sentiment analysis model has been trained on a dataset of tweets. It preprocesses the input text by:

- Converting text to lowercase
- Removing punctuation and emojis
- Stemming and removing stopwords

The model outputs a probability score indicating the sentiment, where a score of 0.5 or higher is classified as positive.

## Contributing

If you'd like to contribute to this project, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) - For creating the web app framework.
- [TensorFlow](https://www.tensorflow.org/) - For the deep learning model.
- [NLTK](https://www.nltk.org/) - For natural language processing tools.
