import streamlit as st
import joblib


# -------------------------
# Page Configuration
# -------------------------

st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="📧",
    layout="wide"
)


# -------------------------
# CSS Design
# -------------------------

st.markdown("""
<style>

.header {

background: linear-gradient(
90deg,
#2563EB,
#7C3AED
);

padding:35px;
border-radius:20px;
text-align:center;
color:white;

}


.header h1 {

font-size:45px;

}


.card {

background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 5px 20px rgba(0,0,0,0.1);

}


.stButton button {

width:100%;
height:50px;

background:#2563EB;

color:white;

font-size:18px;

border-radius:10px;

}


</style>

""", unsafe_allow_html=True)



# -------------------------
# Load Backend Model
# -------------------------

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/spam_model.pkl"
    )

    vectorizer = joblib.load(
        "models/tfidf_vectorizer.pkl"
    )

    return model, vectorizer



model, vectorizer = load_model()



# -------------------------
# Header
# -------------------------

st.markdown("""
<div class="header">

<h1>📧 AI Spam Detection System</h1>

<h3>
Machine Learning Email & SMS Security Analyzer
</h3>

</div>

""",
unsafe_allow_html=True)



st.write("")



# -------------------------
# Sidebar
# -------------------------

with st.sidebar:


    st.title("⚙️ System Info")


    st.write(
    """
    ### Model

    Algorithm:
    **Multinomial Naive Bayes**

    Text Processing:
    **TF-IDF**

    Status:
    🟢 Model Loaded
    """
    )



# -------------------------
# Input
# -------------------------

st.subheader(
    "📝 Enter Email / SMS"
)


message = st.text_area(

    "",

    height=220,

    placeholder=
    "Paste your message here..."

)



# -------------------------
# Prediction
# -------------------------

if st.button("🔍 Analyze Message"):


    if message.strip()=="":


        st.warning(
            "Please enter a message"
        )


    else:


        # Convert text to vector

        vector = vectorizer.transform(
            [message]
        )


        # Prediction

        prediction = model.predict(
            vector
        )[0]


        # Probability

        probability = model.predict_proba(
            vector
        )[0]


        not_spam = probability[0]*100

        spam = probability[1]*100



        st.divider()



        # Result cards

        col1,col2 = st.columns(2)



        with col1:


            if prediction == 1:

                st.error(
                    "🚨 SPAM DETECTED"
                )

            else:

                st.success(
                    "✅ NOT SPAM"
                )



        with col2:


            confidence = max(
                spam,
                not_spam
            )


            st.metric(

                "Confidence",

                f"{confidence:.2f}%"

            )



        st.divider()



        # Probability

        st.subheader(
            "📊 Probability Analysis"
        )


        st.write(
            f"Spam Probability: {spam:.2f}%"
        )


        st.progress(
            int(spam)
        )



        st.write(
            f"Not Spam Probability: {not_spam:.2f}%"
        )


        st.progress(
            int(not_spam)
        )



        st.divider()



        # Risk Level


        st.subheader(
            "⚠️ Risk Level"
        )


        if spam >= 90:

            st.error(
                "🔴 Very High Risk"
            )


        elif spam >= 70:

            st.warning(
                "🟠 High Risk"
            )


        elif spam >= 40:

            st.info(
                "🟡 Suspicious"
            )


        else:

            st.success(
                "🟢 Safe Message"
            )