# README

This repository contains code and data for job description classification experiments, focusing on data processing and transformer-based model training to improve classification accuracy. The project involves two classification setups: 81 classes and 349 classes, each using different data preprocessing methods and transformer models.

## Repository Structure

### Directory for 81-Class Classification

This part of the project involves experiments for classifying job descriptions into 81 categories. The directory is divided into three sections to test different hypotheses:

- **delete_common_words**: This experiment tests the hypothesis that removing frequently occurring but uninformative words can reduce noise and improve classification accuracy.

  Words removed during preprocessing are grouped as follows:

  - **general_words**: Common words that add little context (e.g., "работа" (work), "опыт" (experience), "знание" (knowledge), "компания" (company)).
  - **domain_repeated_words**: Domain-specific words that appear across most job descriptions, such as "продажа" (sales), "оборудование" (equipment), and "документация" (documentation).
  - **generic_adjectives**: Frequently used adjectives, e.g., "высокий" (high), "грамотный" (competent), "ответственность" (responsibility).
  - **time_quantity_words**: Time- and quantity-related words, e.g., "год" (year), "данные" (data), "база" (database).
  - **common_verbs**: Common verbs found in job descriptions, e.g., "создание" (creation), "ведение" (management), "учет" (accounting).
  - **redundant_concepts**: Redundant terms, such as "материал" (material) and "планирование" (planning).
  - **noise**: Specific noise words or symbols, such as "ms".

- **with_prefix**: This experiment adds a prefix with the job title to the start of the job description. The hypothesis is that adding the profession title helps the model classify more accurately by providing context at the beginning.

- **without_prefix**: Here, the model is trained on raw data without removing common words or adding prefixes. This serves as a control dataset for comparing results with other preprocessing techniques.

### Directory for 349-Class Classification

This section involves classifying job descriptions into 349 classes, a more complex task. Three different methods were tested for extracting the main information from job descriptions:

1. **all-mpnet-base-v2**: This method showed insufficient classification quality for the current dataset.
2. **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2**: This method demonstrated good quality and high processing speed. While slightly less accurate than **Flowise Gemma2**, its speed made it the optimal choice.

3. **Flowise Gemma2**: Accessed via the **Flowise** API, this method provided the best accuracy but required significantly more time for processing.

For 349-class classification, **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** was selected as a balanced approach in terms of quality and speed.

### BERT-Based Classification

For 349-class classification, the `DeepPavlov/rubert-base-cased` BERT model is used.

Class labels are encoded as numerical values using `LabelEncoder`:

```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['class'] = encoder.fit_transform(df['class'])
```

The DeepPavlov/rubert-base-cased BERT model was chosen for its capability to handle Russian text and its effective performance on classification tasks.

Dependencies

Install the following libraries to work with this repository:

```pip install pandas
pip install transformers
pip install torch
pip install natasha
pip install tqdm
pip install matplotlib
pip install scikit-learn
pip install nltk```

Experiment Descriptions and Results

Each hypothesis focuses on enhancing job description classification accuracy. The experiments cover various approaches to text preprocessing and classification, with results organized in each subfolder.

    Removing Common Words: delete_common_words — Reducing data noise by removing frequently occurring words to emphasize unique terms.

    Adding Prefixes with Job Titles: with_prefix — Adding prefixes to improve classification accuracy by providing additional context.

    Control Group: without_prefix — Training on raw data without preprocessing for a baseline comparison.

    349-Class Classification Using Information Extraction Methods: The optimal method was sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, a balanced choice in terms of quality and speed.
