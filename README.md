# 🤖 Next Word Prediction using LSTM

![Python](https://img.shields.io/badge/Python-3.10-blue.svg) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg)

This project is a simple implementation of a next word prediction model using a Long Short-Term Memory (LSTM) network. The model is trained on a custom dataset of FAQs to generate text that mimics the style of the source text.

---

### ✨ Features

-   **Text Preprocessing**: Cleans and tokenizes the input text data.
-   **Sequence Generation**: Creates n-gram sequences for model training.
-   **LSTM Model**: A simple Keras-based LSTM model to learn text patterns.
-   **Text Generation**: Predicts and generates a sequence of words based on a starting phrase.

---

### 演示 (Demo)

Here's a quick look at the model in action after training:

![Image](https://github.com/user-attachments/assets/c1333d8e-0d66-4941-9497-db1d1654f4c3)
---

### 🛠️ Tech Stack

-   **Python 3.10**
-   **TensorFlow 2.12**
-   **NumPy 1.23**
-   **Keras** (as part of TensorFlow)

---

### ⚙️ Setup and Installation

Follow these steps to set up the project locally.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/jsonusuman351/Nextwordpredict.git](https://github.com/jsonusuman351/Nextwordpredict.git)
    cd Nextwordpredict
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create the venv using Python 3.10
    conda create -p venv python==3.10.0


    # Activate it
    conda activate venv/
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

---

### 🚀 Usage

Once the setup is complete, you can run the project.

1.  **Train the Model:**
    This script will preprocess the data, build the LSTM model, train it, and save the trained model and tokenizer in the `saved_models/` directory.
    ```bash
    python src/train.py
    ```

2.  **Generate Text:**
    After training, run this script to see the predictions. You can change the starting text inside the `predict.py` file.
    ```bash
    python src/predict.py
    ```
    **Example Output:**
    ```
    --------------------------------------------------
    Seed Text: 'what is the duration'
    Generated Text: 'what is the duration of the course is 7 months so the total course'
    --------------------------------------------------
    ```

---

### 📂 Project Structure

<details>
<summary>Click to view the folder structure</summary>

```
Nextwordpredict/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   └── faqs.txt
│
├── saved_models/
│   └── .gitkeep
│
└── src/
    ├── __init__.py
    ├── data_preprocessing.py
    ├── model.py
    ├── train.py
    └── predict.py
```
</details>

---

### 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---
