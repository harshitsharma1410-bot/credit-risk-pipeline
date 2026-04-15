# 🏦 Credit Risk Predictive Risk Engine

## Overview
The Credit Risk Predictive Risk Engine is a hybrid Machine Learning and Generative AI pipeline designed for the financial sector. It evaluates loan applications, dynamically prices risk, enforces banking regulations, and generates highly personalized, human-readable customer communications using LLMs.

Unlike standard classification models, this engine utilizes a **5-stage architecture** to mimic a real-world underwriting environment, combining the mathematical rigor of XGBoost with the empathy and contextual awareness of Llama 3.

## 🧠 Core Architecture

1. **The Guardrails (Business Rules Engine - BRE):** - Enforces strict banking regulations before AI evaluation.
   - Triggers Auto-Rejections for zero-income or prior default histories assuming its recent but since in the data used we are don’t have any credit history time lines we have assumed all the defaulters were recently defaulted for the goodwill of this project.
   - Features a **Tailoring Engine** that safely scales down unrealistic loan requests based on Debt-to-Income (DTI) maximums (e.g., capping Home Loans at 5x income).
2. **The Pricing Engine (Smart Imputation):** - Utilizes an `XGBRegressor` to calculate a dynamic, market-accurate interest rate based on the applicant's financial profile.
   - Applies mathematical "clamps" to ensure rates stay within legal government buckets (e.g., Home Loans bounded between 7.10% and 10.50%).
3. **The Risk Engine (Classification):** - A hyper-tuned `XGBClassifier` evaluates the tailored application to predict the probability of default.
   - Evaluated against an optimized F1-Score threshold to balance risk aversion with business growth. By optimized F1 score means we have 
4. **The Context Assembly:** - Formats raw mathematical outputs into clean, user-friendly variables (e.g., formatting `8000000` to `₹80,00,000`).
5. **The Empathy Layer (Generative AI):** - Integrates with local **Ollama (Llama 3)** via API.
   - Translates the mathematical decisions into warm, compliant, and professional email copy. Includes logic to seamlessly pitch "Conditional Approvals" if the loan amount was tailored down.

## ⚙️ Model Tuning & Mathematics

To ensure production-grade accuracy and business viability, the models were mathematically calibrated beyond standard defaults:

* **Objective & Loss Functions:**
  * **Risk Engine (Classifier):** Optimized This heavily penalizes the model for being confidently wrong, ensuring the output probabilities are highly calibrated for financial risk assessment.
  * **Pricing Engine (Regressor):** Optimized using ** to fiercely penalize large deviations when predicting the correct market interest rate.
* **Imbalance Handling:** The dataset naturally contains fewer defaults than safe loans. This was handled dynamically by calculating the imbalance ratio and passing it to XGBoost's `scale_pos_weight`, forcing the model to pay strict mathematical attention to the minority class.
* **Business-Driven Threshold Optimization:** Rather than accepting the default `0.5` decision boundary, the engine evaluates the **Precision-Recall Curve** on the test data. It mathematically identifies and applies the exact probability threshold that maximizes the **F1-Score**, perfectly balancing the bank's need to catch defaults against the goal of approving safe loans.


## 🛠️ Tech Stack
* **Data Processing:** Python, Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost (`XGBClassifier`, `XGBRegressor`)
* **Generative AI:** Ollama (Local Llama 3)
* **API / Networking:** `requests`

## 🚀 Getting Started

### Prerequisites
1. Install Python 3.9+
2. Install necessary pip packages: `pip install pandas numpy xgboost scikit-learn requests`
3. Download and install [Ollama](https://ollama.com/).
4. Pull the Llama 3 model by running this in your terminal:
   ```bash
   ollama run llama3


### 📊 Data Setup

Because of size limits and privacy best practices, the raw `.csv` data is not included in this repository. To run this project locally:

1. Download the [credit_risk_dataset.csv](https://www.kaggle.com/datasets/laotse/credit-risk-dataset?resource=download&select=credit_risk_dataset.csv) file from Kaggle.
2. Create a folder named `data` in the root directory of this project.
3. Place the downloaded `.csv` file inside the `data` folder.
