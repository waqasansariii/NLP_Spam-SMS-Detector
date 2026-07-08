import streamlit as st
import nltk
import os
import joblib
import re
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
for pkg in ['punkt', 'punkt_tab', 'stopwords']:
    try:
        nltk.download(pkg, quiet=True)
    except:
        pass

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_word= set(stopwords.words('english'))
stemmer = PorterStemmer()

# Preprocess Function


st.set_page_config(
    page_title="NLP EMAIL SPAM DETECTOR",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>

/* Main App */
.main {
    padding-top: 2rem;
}

/* Title */
.main-title{
    font-size:42px;
    font-weight:bold;
    text-align:center;
    color:#1f77b4;
}

/* Subtitle */
.sub-title{
    text-align:center;
    font-size:18px;
    color:gray;
    margin-bottom:30px;
}

/* Result Card */
.result-card{

    padding:20px;

    border-radius:15px;

    background-color:#F5F5F5;

    box-shadow:0px 2px 10px rgba(0,0,0,0.2);

    text-align:center;

    margin-top:20px;

}

</style>
""", unsafe_allow_html=True)

@st.cache_resource

def load_model():
    try:
        model = joblib.load("spam_detector.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'spam_detector.pkl' and 'vectorizer.pkl' are in the project directory.")
        st.stop()
        

model, vectorizer = load_model()

def preproces_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_word]  # Remove stopwords
    tokens = [stemmer.stem(word) for word in tokens]  # Stemming
    return ' '.join(tokens)

def predict_sms(message):
    cleaned_text = preproces_text(message)
    vector = vectorizer.transform([cleaned_text])   
    prediction = model.predict(vector)
    probabilities = model.predict_proba(vector)[0]
    return prediction, probabilities

st.title("📧 Spam SMS Detector")

st.markdown("""
<p class='sub-title'>
Detect Spam Messages using NLP
</p>
""", unsafe_allow_html=True)
st.sidebar.title("📊 Model Information")
st.sidebar.markdown("### 🤖 Algorithm")
st.sidebar.write("Multinomial Naive Bayes")
st.sidebar.markdown("### 📝 Vectorizer")
st.sidebar.write("TF-IDF")
st.sidebar.markdown("### 📂 Dataset")
st.sidebar.write("UCI SMS Spam Collection")
st.sidebar.markdown("### 📈 Model Performance")

st.sidebar.write("Accuracy : 96.65%")
st.sidebar.write("Precision : 98.10%")
st.sidebar.write("Recall :  74.04%")
st.sidebar.write("F1 Score : 84.71%")
st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 Developer")
st.sidebar.write("Waqas Ansari")

st.sidebar.markdown("### 🛠 Tech Stack")
st.sidebar.write("""
- Python
- Streamlit
- Scikit-learn
- NLTK
- TF-IDF
- Joblib
""")

user_input = st.text_area(
    "Enter your SMS",
    height=150,
    placeholder="Type your message here..."
)
col1,col2,col3 = st.columns([1,2,1])
with col2:
    predict_button = st.button(
        "🔍 Analyze Message",
        use_container_width=True
    )
if predict_button:
    if user_input.strip()=="":
        st.warning("Please enter a message to analyze.")
    else:
        prediction, probabilities = predict_sms(user_input)
        st.write(prediction, probabilities)
        if prediction == 1:
            confidence = probabilities[1] * 100
            st.markdown(f"""<div class="result-card">
<h2>🚨 Spam Message</h2>

<h3>Confidence : {confidence:.2f}%</h3>

</div>
""", unsafe_allow_html=True)
        else:
            confidence = probabilities[0] * 100
            st.markdown(f"""<div class="result-card">

<h2>✅ Ham Message</h2>

<h3>Confidence : {confidence:.2f}%</h3>

</div>
""", unsafe_allow_html=True)
        st.write("### Prediction Probabilities")
        st.write(f"📩 Ham : {probabilities[0]*100:.2f}%")
        st.write(f"🚨 Spam : {probabilities[1]*100:.2f}%")
        st.write("Ham Probability")
        st.progress(float(probabilities[0]))
        st.write("Spam Probability")
        st.progress(float(probabilities[1]))