# Olist E-Commerce Analysis

Brazilian E-Commerce analysis using the Olist public dataset — EDA, RFM customer segmentation & delivery delay prediction.

## Dataset

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — ~100k orders placed between 2016 and 2018, across multiple CSV files covering orders, products, payments, customers, sellers and reviews.

## What this project does

### 1. EDA
- Monthly revenue trend (2016–2018)
- Top 10 product categories by revenue

### 2. RFM Customer Segmentation
Customers are scored on three dimensions:
- **Recency** — how recently they purchased
- **Frequency** — how many orders they placed
- **Monetary** — how much they spent

Segments: Champions, Loyal, New Customers, Potential, At Risk, Lost.

### 3. Delivery Delay Prediction
A Random Forest classifier predicts whether an order will arrive late, using features like approval delay, estimated delivery days, and number of items.

## Setup

**1. Install dependencies:**
```bash
pip install pandas matplotlib seaborn scikit-learn
```

**2. Download the dataset** from Kaggle and place the CSV files in a folder named `db/`.

**3. Run:**
```bash
python main.py
```

## Output

Three plots saved as PNG files:

| File | Content |
|---|---|
| `1_eda.png` | Monthly revenue + top categories |
| `2_rfm.png` | RFM segmentation pie chart & scatter |
| `3_feature_importance.png` | Feature importance for delay prediction |

## Project Structure

```
.
├── db/                          # CSV files from Kaggle (not included)
├── main.py                      # Main analysis script
├── 1_eda.png                    # Output
├── 2_rfm.png                    # Output
├── 3_feature_importance.png     # Output
└── README.md
```

## Tech Stack

- **Python 3.11**
- pandas, matplotlib, seaborn, scikit-learn
