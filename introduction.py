import streamlit as st

# Page configuration
st.set_page_config(page_title="Introduction", page_icon=":guardsman:", layout="wide")

# Title
st.title("Introduction to Movie Prediction ML Models")

# Creator Information Section
st.header("Creators")
creator_info = """
- กษิดิส ทองบุญ (ID: 6604062630030)
- ณัฐพงศ์ จันทร์เพ็ง (ID: 6604062630188)
"""
st.markdown(creator_info)

# Section Information
st.subheader("Section: 3")

# Models Information Section
st.header("Models Used")
models_info = """
- **Machine Learning** (Random Forest)
- **Machine Learning** (XGBoost)
- **Neural Network** (MLP - Multi-layer Perceptron)
"""
st.markdown(models_info)

# References Section
st.header("References")
references_info = """
- Big thanks to **[ChatGPT](https://chat.openai.com)**
- [Kaggle - [Customer Churn Dataset](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset?resource=download)
- [Random Forest Wikipedia page](https://en.wikipedia.org/wiki/Random_forest)
- [XGBoost Wikipedia page](https://en.wikipedia.org/wiki/XGBoost)
- [TensorFlow Documentation](https://www.tensorflow.org/?hl=th)
- [Kaggle - Sample34](https://www.kaggle.com/datasets/jacksondivakarr/sample34)
- [Understanding MLPs - GeeksforGeeks](https://www.geeksforgeeks.org/multi-layer-perceptron-learning-in-tensorflow/)
"""
st.markdown(references_info)

# Styling and Layout Customization
st.markdown(
    """
    <style>
    .main { 
        font-family: 'Arial', sans-serif; 
        background-color: #f4f7fa;
        padding: 10px;
    }
    h1, h2, h3 {
        color: #333;
    }
    .markdown-text-container {
        padding: 10px;
        border-radius: 5px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True
)
