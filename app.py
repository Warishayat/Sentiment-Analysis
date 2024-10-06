import streamlit  as st
import pickle
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import warnings
import string

warnings.filterwarnings("ignore")
extract = string.punctuation
nltk.download('stopwords')
nltk.download('wordnet')

#load the model that save embedding and model trained file.
Model = pickle.load(open("model.pkl","rb"))
Tokenized = pickle.load(open("token.pkl","rb"))

#preprocessing:
def Data_Preprocessing(text):
  text=text.lower()  #convert into lower case

  text=text.translate(str.maketrans('','',extract)) #remove punctuation

  data = re.compile('<.*?>')
  text = re.sub(data, "", text)


  new_text = []
  for i in text.split():
    if i not in stopwords.words('english'):  #for remove stopwords
      new_text.append(i)
  text = " ".join(new_text)


  emojies_pattren = re.compile("["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
                                )
  text = re.sub(emojies_pattren, "", text)  #for removing the emojies


  #stemming
  ps = PorterStemmer()
  stem_text = []
  for i in text.split():
    stem_text.append(ps.stem(i))
  text = " ".join(stem_text)

  word = text.split()
  n_text = []
  for w in word:
    if not  w.isdigit():
      n_text.append(w)
  text = " ".join(n_text)

  return text


#Model_Embeddings

def Tokenized_text(text):
    text = pad_sequences(Tokenized.texts_to_sequences(text), maxlen=200)
    return text

#Model prediction will be here
def Model_prediction(text):
    # Predict the probabilities
    result = Model.predict(text)
    return result



st.title("Sentiment Analysis App (NLP)")

col1,col2 = st.columns(2)

with col1:
    st.text("""Sentiment Analysis is a technique that
determines the emotional tone of text,
classifying it as positive, negative,or 
neutral.It provides insights into customer
opinions,aiding businesses in decision-
making and improving engagement through 
data analysis.""")

with col2:
    st.image("sentiment_analysis.jfif")

tweet=st.text_input("Enter the tweet want you check Sentiment")

button_predict = st.button("Predict")

if button_predict:
    text = Data_Preprocessing(tweet)
    text = Tokenized_text([text])  # Ensure it's a list
    result = Model_prediction(text)

    sentiment_probability = result[0][0]  # Get the first prediction value

    # Print the raw probability for debugging
    st.write(f"Raw prediction probability: {sentiment_probability}")

    # Determine sentiment based on the probability
    if sentiment_probability >= 0.5:
        st.text("Entered Sentiment is Positive")
    else:
        st.text("Entered Sentiment is Negative")






