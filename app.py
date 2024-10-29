import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt_tab')
nltk.download('stopwords')
# making a function which include all the steps
# Download punkt resource if it hasn't been downloaded yet
# # try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

def transform_text(text):
    # lowercase
    text = text.lower()

    # tokensization
    text = nltk.word_tokenize(text)

    # removing special characters and taking only alphanumeric characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # removing stopwords and puncuations
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # stemming
    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Conversation Analyzer")

input_text = st.text_area("Enter the conversation/msg/email/calltranscript")

# 1. preprocess the text

if st.button('Predict'):


    transformed_message = transform_text(input_text)

    # 2. vectorize

    vector_input = tfidf.transform([transformed_message])

    # 3. predict

    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.header("Spam/Fraud")
        st.button("Change into not fraud")
    else :
        st.header("Not Spam/ Not Fraud")
        st.button("Change into fraud")

