# Human vs AI Essays Dashboard

## Overview
Python project that analyzes and classifies essays as human-written or AI-generated. 
Includes a full data pipeline for text preprocessing and feature extraction, and an interactive **Streamlit dashboard** for visualization and prediction.

## Dataset
The dataset of human vs AI essays is included in this repo.  
**License:** CC0 (Public Domain)  
Source: [Kaggle](https://www.kaggle.com/datasets/navjotkaushal/human-vs-ai-generated-essays)

## Features
- Text preprocessing with **NLTK** (stopwords removal, cleaning)
- Feature extraction (word count, unique words, vocabulary diversity)
- Machine learning classification for human vs AI essays
- Interactive **Streamlit dashboard**:
  - Visualize dataset statistics and distributions
  - Input an essay for prediction

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Dkerker/human-vs-ai-essays.git
cd human-vs-ai-essays
```
2. Create and activate virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Run Streamlit dashboard
```bash
streamlit run src/app.py
```

## Technologies
- Python, Pandas, Numpy, NLTK
- Scikit-learn
- Streamlit, Plotly
- Git & Github
