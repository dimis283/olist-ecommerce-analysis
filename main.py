"""
=============================================================
  Olist Brazilian E-Commerce — Demo Analysis
  Requirements: pip install pandas matplotlib seaborn scikit-learn

  Structure:
    1. Load & overview of data
    2. EDA — Monthly revenue, top categories
    3. RFM customer segmentation
    4. ML — Delivery delay prediction
=============================================================
  Download the dataset from:
  https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
  and place the CSV files inside a folder named "db/".
=============================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "db"  # ← change this if your folder has a different name/path

# ── Styling ──────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120


# =============================================================
# 1. LOAD DATA
# =============================================================
print("=" * 55)
print("  Olist E-Commerce Demo")
print("=" * 55)
print("\n📂 Loading files...")

orders       = pd.read_csv(os.path.join(DATA_DIR, "olist_orders_dataset.csv"), parse_dates=[
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
])
order_items  = pd.read_csv(os.path.join(DATA_DIR, "olist_order_items_dataset.csv"))
products     = pd.read_csv(os.path.join(DATA_DIR, "olist_products_dataset.csv"))
categories   = pd.read_csv(os.path.join(DATA_DIR, "product_category_name_translation.csv"))
payments     = pd.read_csv(os.path.join(DATA_DIR, "olist_order_payments_dataset.csv"))
customers    = pd.read_csv(os.path.join(DATA_DIR, "olist_customers_dataset.csv"))

print(f"  ✔ Orders:      {len(orders):,}")
print(f"  ✔ Order Items: {len(order_items):,}")
print(f"  ✔ Products:    {len(products):,}")
print(f"  ✔ Customers:   {len(customers):,}")


# =============================================================
# 2. EDA
# =============================================================
print("\n📊 EDA...")

# — Monthly revenue —
delivered = orders[orders["order_status"] == "delivered"].copy()
revenue = (
    delivered
    .merge(payments[["order_id", "payment_value"]], on="order_id")
    .merge(order_items[["order_id"]].drop_duplicates(), on="order_id")
)
revenue["month"] = revenue["order_purchase_timestamp"].dt.to_period("M")
monthly = revenue.groupby("month")["payment_value"].sum().reset_index()
monthly["month_str"] = monthly["month"].astype(str)

# — Top 10 categories by revenue —
items_with_cat = (
    order_items
    .merge(products[["product_id", "product_category_name"]], on="product_id")
    .merge(categories, on="product_category_name", how="left")
)
items_with_cat["cat"] = items_with_cat["product_category_name_english"].fillna(
    items_with_cat["product_category_name"]
)
top_cats = (
    items_with_cat.groupby("cat")["price"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Olist — Sales Overview", fontsize=14, fontweight="bold")

# subplot 1: monthly revenue
ax1 = axes[0]
ax1.bar(monthly["month_str"], monthly["payment_value"] / 1e6, color="#4C72B0")
ax1.set_title("Monthly Revenue (M BRL)")
ax1.set_xlabel("")
ax1.set_ylabel("Revenue (M BRL)")
ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
step = max(1, len(monthly) // 8)
ax1.set_xticks(range(0, len(monthly), step))
ax1.set_xticklabels(monthly["month_str"].iloc[::step], rotation=45, ha="right")

# subplot 2: top categories
ax2 = axes[1]
sns.barplot(data=top_cats, y="cat", x="price", ax=ax2, color="#DD8452")
ax2.set_title("Top 10 Categories (Revenue)")
ax2.set_xlabel("Revenue (BRL)")
ax2.set_ylabel("")
ax2.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

plt.tight_layout()
plt.savefig("1_eda.png")
plt.show()
print("  ✔ Saved: 1_eda.png")


# =============================================================
# 3. RFM SEGMENTATION
# =============================================================
print("\n🧩 RFM Segmentation...")

snapshot_date = orders["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

rfm_base = (
    delivered
    .merge(payments[["order_id", "payment_value"]], on="order_id")
    .merge(customers[["customer_id", "customer_unique_id"]], on="customer_id")
)

rfm = rfm_base.groupby("customer_unique_id").agg(
    Recency   = ("order_purchase_timestamp", lambda x: (snapshot_date - x.max()).days),
    Frequency = ("order_id",                 "nunique"),
    Monetary  = ("payment_value",             "sum"),
).reset_index()

# Score each dimension 1–5
# Robust version: adjusts labels if qcut produces fewer than 5 bins due to duplicates
def safe_qcut(series, q=5, ascending=True):
    tmp = pd.qcut(series, q=q, labels=False, duplicates="drop")
    n_bins = tmp.nunique()
    labels = list(range(1, n_bins + 1)) if ascending else list(range(n_bins, 0, -1))
    return pd.qcut(series, q=q, labels=labels, duplicates="drop")

for col in ["Recency", "Frequency", "Monetary"]:
    ascending = col != "Recency"  # Recency: lower value = better score
    rfm[f"{col}_Score"] = safe_qcut(rfm[col], ascending=ascending)

rfm["RFM_Score"] = (
    rfm["Recency_Score"].astype(str)
    + rfm["Frequency_Score"].astype(str)
    + rfm["Monetary_Score"].astype(str)
)

def segment(row):
    r, f, m = int(row["Recency_Score"]), int(row["Frequency_Score"]), int(row["Monetary_Score"])
    if r >= 4 and f >= 4:
        return "Champions"
    elif r >= 3 and f >= 3:
        return "Loyal"
    elif r >= 4 and f <= 2:
        return "New Customers"
    elif r <= 2 and f >= 3:
        return "At Risk"
    elif r <= 2 and f <= 2:
        return "Lost"
    else:
        return "Potential"

rfm["Segment"] = rfm.apply(segment, axis=1)

seg_counts  = rfm["Segment"].value_counts()
seg_revenue = rfm.groupby("Segment")["Monetary"].mean().round(2)

print("\n  Segment breakdown:")
print(pd.DataFrame({"Customers": seg_counts, "Avg Revenue (BRL)": seg_revenue}).to_string())

palette = {
    "Champions": "#2ecc71", "Loyal": "#3498db", "New Customers": "#9b59b6",
    "Potential": "#f39c12", "At Risk": "#e67e22", "Lost": "#e74c3c",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("RFM Customer Segmentation", fontsize=14, fontweight="bold")

# pie chart: customer distribution per segment
seg_df = seg_counts.reset_index()
seg_df.columns = ["Segment", "Count"]
colors = [palette[s] for s in seg_df["Segment"]]
axes[0].pie(seg_df["Count"], labels=seg_df["Segment"], autopct="%1.1f%%",
            colors=colors, startangle=140)
axes[0].set_title("Customer % per Segment")

# scatter: recency vs monetary colored by segment
sns.scatterplot(
    data=rfm.sample(min(3000, len(rfm)), random_state=42),
    x="Recency", y="Monetary", hue="Segment",
    palette=palette, alpha=0.5, ax=axes[1], s=15,
)
axes[1].set_title("Recency vs Monetary")
axes[1].set_xlabel("Recency (days)")
axes[1].set_ylabel("Monetary (BRL)")
axes[1].legend(title="Segment", bbox_to_anchor=(1.01, 1), loc="upper left")

plt.tight_layout()
plt.savefig("2_rfm.png")
plt.show()
print("  ✔ Saved: 2_rfm.png")


# =============================================================
# 4. ML — DELIVERY DELAY PREDICTION
# =============================================================
print("\n🤖 ML: Delivery delay prediction...")

# Keep only delivered orders with complete date info
ml = delivered.dropna(subset=[
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "order_approved_at",
    "order_purchase_timestamp",
]).copy()

# Target: 1 if order arrived after the estimated date
ml["is_late"] = (
    ml["order_delivered_customer_date"] > ml["order_estimated_delivery_date"]
).astype(int)

# Feature: hours between purchase and payment approval
ml["approval_delay"] = (
    ml["order_approved_at"] - ml["order_purchase_timestamp"]
).dt.total_seconds() / 3600

# Feature: estimated delivery window in days
ml["estimated_days"] = (
    ml["order_estimated_delivery_date"] - ml["order_purchase_timestamp"]
).dt.days

# Features: time of purchase
ml["purchase_month"]     = ml["order_purchase_timestamp"].dt.month
ml["purchase_dayofweek"] = ml["order_purchase_timestamp"].dt.dayofweek

# Feature: number of items per order
item_counts = order_items.groupby("order_id")["order_item_id"].count().reset_index()
item_counts.columns = ["order_id", "n_items"]
ml = ml.merge(item_counts, on="order_id", how="left")
ml["n_items"] = ml["n_items"].fillna(1)

features = ["approval_delay", "estimated_days", "purchase_month",
            "purchase_dayofweek", "n_items"]
ml_clean = ml[features + ["is_late"]].dropna()

X = ml_clean[features]
y = ml_clean["is_late"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

print("\n  Classification Report:")
print(classification_report(y_test, clf.predict(X_test),
                             target_names=["On Time", "Late"]))

# Plot feature importance
importance = pd.DataFrame({
    "Feature":    features,
    "Importance": clf.feature_importances_,
}).sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(importance["Feature"], importance["Importance"], color="#4C72B0")
ax.set_title("Feature Importance — Delivery Delay Prediction", fontweight="bold")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("3_feature_importance.png")
plt.show()
print("  ✔ Saved: 3_feature_importance.png")


# =============================================================
print("\n" + "=" * 55)
print("  ✅ Done!")
print("  Output: 1_eda.png | 2_rfm.png | 3_feature_importance.png")
print("=" * 55)
