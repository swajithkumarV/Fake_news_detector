# 📰 Fake News Detector using Machine Learning

An end-to-end Fake News Detection system built using Python, Machine Learning, Natural Language Processing (NLP), and Streamlit.

This project uses **TF-IDF vectorization and Logistic Regression** to analyze news text and classify it as Real News or Fake News.

## 🚀 Features

- 🔍 Predict whether a news article is Real or Fake.
- 🤖 Machine Learning model using Logistic Regression.
- 📝 Text preprocessing and TF-IDF feature extraction.
- 📊 Prediction probabilities displayed by the application.
- 🌐 Interactive web interface built with Streamlit.
- 📦 Trained ML pipeline saved for reuse.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF Vectorizer
- Logistic Regression
- Streamlit
- Joblib
- Jupyter Notebook

## 📂 Project Structure

```text
Fake_news_detector/
├── app.py
├── final_model_pipeline.pkl
├── news_prdict.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/swajithkumarV/Fake_news_detector.git
```

### 2. Navigate to the project folder

```bash
cd Fake_news_detector
```

### 3. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will provide a local URL in your terminal. Open it in your browser to use the detector.

## 🧠 How It Works

1. The user enters news text into the Streamlit application.
2. The trained pipeline converts the text into numerical features using TF-IDF.
3. Logistic Regression predicts the class.
4. The application displays the predicted category and, if configured, prediction probabilities.

## 📊 Dataset

This project uses the **WELFake dataset**, a dataset of real and fake news articles used for training and evaluating fake news classification models.

The dataset file is not included in this repository.

## 🎯 Project Objective

The objective is to demonstrate how Machine Learning and NLP can be applied to text classification and help users examine potentially misleading news content.

## ⚠️ Limitations

This model is not a definitive fact-checking system. Its predictions can be incorrect, so important news claims should be verified using reliable sources.

## 👨‍💻 Author

**Swajith Kumar**

GitHub: [@swajithkumarV](https://github.com/swajithkumarV)

---

⭐ If you find this project interesting, consider giving the repository a star.
