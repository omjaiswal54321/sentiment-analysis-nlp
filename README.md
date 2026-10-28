# Sentiment Analysis Using Machine Learning and Deep Learning

A comparative Natural Language Processing (NLP) project that evaluates four approaches to classifying text sentiment: TF-IDF with Logistic Regression, a PyTorch neural network, fine-tuned DistilBERT, and an API-based language model.

The project classifies text into three sentiment categories: **Negative, Neutral, and Positive**.

## Overview

The goal of this project is to explore how different machine learning and deep learning approaches perform on a multiclass sentiment classification task.

The project covers data cleaning, text preprocessing, model development, evaluation, and performance comparison.

### Models Implemented

- **TF-IDF + Logistic Regression:** A traditional machine learning baseline.
- **TF-IDF + PyTorch Neural Network:** A feed-forward neural network trained on TF-IDF features.
- **DistilBERT:** A pretrained Transformer fine-tuned for sentiment classification.
- **Jev API:** An API-based language model evaluated on the sentiment classification task.

## Dataset

The repository includes a cleaned dataset named `test.csv`, containing **14,402 text samples** distributed across three sentiment classes.

| Sentiment | Number of Samples |
|---|---:|
| Negative | 9,079 |
| Neutral | 3,046 |
| Positive | 2,277 |
| **Total** | **14,402** |

The dataset is imbalanced, with negative sentiment representing the largest class. Therefore, the evaluation considers macro F1-score alongside accuracy and weighted F1-score.

### Data Cleaning and Preprocessing

The data-cleaning workflow includes:

- Removing records with missing text or sentiment labels.
- Removing text samples with conflicting sentiment labels.
- Removing duplicate text samples.
- Preparing the cleaned data for model training and evaluation.

Text preprocessing differs by approach. The TF-IDF models use processed text, while DistilBERT uses tokenizer-compatible text input. This difference is important when interpreting the model comparisons.

## Model Performance

The following results were recorded in the model evaluation notebooks.

| Model | Accuracy | Macro F1-Score | Weighted F1-Score |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 76.85% | 0.7235 | 0.7746 |
| TF-IDF + PyTorch Neural Network | 78.48% | 0.72 | 0.78 |
| Fine-tuned DistilBERT | 82.68% | 0.79 | 0.83 |
| Jev API | 84.01% | 0.77 | 0.83 |

### Key Findings

- **Highest recorded accuracy:** Jev API, at 84.01% on successfully evaluated examples.
- **Highest macro F1-score:** DistilBERT, at 0.79.
- The PyTorch neural network improved on the Logistic Regression baseline in test accuracy.
- Neutral sentiment was more challenging to classify than negative sentiment.

**Evaluation note:** The Jev API returned predictions for 2,877 of 2,881 test samples. Four examples were blocked by the API's safety mechanism. Its reported metrics use successfully evaluated examples only. A strict comparison requires verifying that the models use comparable test examples and evaluation conditions.

## Tech Stack

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn
- **Deep learning:** PyTorch
- **NLP:** NLTK, TF-IDF, Hugging Face Transformers
- **Transformer model:** DistilBERT
- **Visualization:** Matplotlib
- **Development environment:** Jupyter Notebook / Google Colab
- **API-based evaluation:** Jev API

## Project Structure

```text
sentiment-analysis-nlp/
│
├── README.md
├── requirements.txt
├── test.csv
├── Sentiment_Data_Cleaning_Simple.ipynb
├── 01_TFIDF_Logistic_Regression.ipynb
├── 02_TFIDF_PyTorch_Neural_Network.ipynb
├── 03_Pretrained_Transformer.ipynb
└── 04_Jev_Evaluation.ipynb
```

The dataset is provided in cleaned form, so the cleaning notebook is optional when reproducing the modeling experiments. Run it only if you want to reproduce the data-cleaning process.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/omjaiswal54321/sentiment-analysis-nlp.git
cd sentiment-analysis-nlp
```

### 2. Install Dependencies

Install the required Python libraries using `requirements.txt`:

```bash
pip install -r requirements.txt
```

Additional dependencies may be required depending on the notebook and runtime environment. The requirements file should reflect the libraries needed by the notebooks.

### 3. Prepare the Dataset

The cleaned dataset is provided as `test.csv` in the repository root.

Ensure that the dataset path and filename referenced in each notebook match the file's location.

### 4. Run the Notebooks

Run the modeling notebooks in the following order:

1. `01_TFIDF_Logistic_Regression.ipynb`
2. `02_TFIDF_PyTorch_Neural_Network.ipynb`
3. `03_Pretrained_Transformer.ipynb`
4. `04_Jev_Evaluation.ipynb`

To reproduce the cleaning process, run `Sentiment_Data_Cleaning_Simple.ipynb` separately before the modeling notebooks.

The Jev evaluation requires valid API access and the appropriate configuration. Some model evaluations may require a GPU for practical execution time.

**Important:** The Jev notebook uses checkpoints to resume evaluation and avoid repeating completed API calls. Avoid rerunning API-calling cells unnecessarily, as they may consume API usage.

## Evaluation Metrics

The models are evaluated using:

- **Accuracy:** The proportion of correctly classified examples.
- **Precision:** The proportion of predicted examples for a class that are correct.
- **Recall:** The proportion of actual examples of a class that are identified.
- **Macro F1-score:** The unweighted average of F1-scores across classes.
- **Weighted F1-score:** The average of class F1-scores weighted by class frequency.
- **Confusion matrix:** A breakdown of correct and incorrect predictions for each class.

## Challenges and Limitations

- The dataset has an imbalanced class distribution.
- Neutral sentiment can be difficult to distinguish from positive or negative sentiment.
- Text preprocessing differs between the TF-IDF models and Transformer-based approaches.
- API-based evaluation may have incomplete prediction coverage because of safety filters or service limitations.
- Fair model comparison depends on using consistent test examples and evaluation conditions.

## Future Improvements

- Perform deeper error analysis on neutral sentiment predictions.
- Explore class weighting and other strategies for handling class imbalance.
- Compare models using identical test examples and comparable evaluation conditions.
- Save reusable model artifacts and build a prediction interface.
- Develop a lightweight interactive application using Streamlit.
- Add visualizations for class-wise performance and confusion matrices.

## Conclusion

This project compares traditional machine learning, a feed-forward neural network, a fine-tuned Transformer, and an API-based language model for sentiment classification.

DistilBERT achieved the highest recorded macro F1-score, while Jev achieved the highest recorded accuracy on successfully evaluated examples. The experiments also highlight the difficulty of classifying neutral sentiment in an imbalanced dataset.

The project demonstrates practical experience in NLP preprocessing, model training, deep learning, Transformer fine-tuning, evaluation metrics, and comparative experimentation.

---

**Author:** Om Jaiswal

- GitHub: [omjaiswal54321](https://github.com/omjaiswal54321)
- LinkedIn: [om-jais](https://www.linkedin.com/in/jais-om/)
