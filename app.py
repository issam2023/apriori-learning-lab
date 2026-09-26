import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Apriori Learning Lab", layout="wide")

st.title("Association Rules — Apriori Learning Lab")
st.write("Raw Data → Clean Data → Baskets → Boolean Matrix → Apriori → Frequent Itemsets → Association Rules")

# --------------------------------------------------
# 1. RAW DATA
# --------------------------------------------------
st.header("1. Raw Data")

df = pd.read_csv("grocery_transactions.csv")

st.write("Each row represents one item inside a transaction.")
st.dataframe(df.head(20), width='stretch')

st.write("Rows:", len(df))
st.write("Transactions:", df["transaction_id"].nunique())

# --------------------------------------------------
# 2. CLEAN DATA
# --------------------------------------------------
st.header("2. Clean the Data")

clean_df = df.copy()
clean_df["item"] = clean_df["item"].astype(str).str.strip().str.lower()

st.code(
    'df["item"] = df["item"].str.strip().str.lower()',
    language="python"
)

st.write("We remove extra spaces and convert item names to lowercase.")
st.dataframe(clean_df.head(20), width='stretch')

# --------------------------------------------------
# 3. CREATE BASKETS
# --------------------------------------------------
st.header("3. Create Baskets")

baskets = (
    clean_df.groupby("transaction_id")["item"]
    .apply(lambda items: sorted(set(items)))
    .tolist()
)

basket_ids = clean_df["transaction_id"].drop_duplicates().tolist()

basket_df = pd.DataFrame({
    "transaction_id": basket_ids,
    "basket": baskets
})

st.write("Rows with the same transaction ID are grouped into one basket.")
st.dataframe(basket_df.head(15), width='stretch')

# --------------------------------------------------
# 4. BOOLEAN MATRIX
# --------------------------------------------------
st.header("4. Boolean Matrix — TransactionEncoder")

te = TransactionEncoder()
encoded = te.fit(baskets).transform(baskets)

basket_matrix = pd.DataFrame(
    encoded,
    columns=te.columns_
)

display_matrix = basket_matrix.astype(int)
display_matrix.index = basket_ids

st.write("1 = item is present, 0 = item is absent.")
st.dataframe(display_matrix.head(15), width='stretch')

# --------------------------------------------------
# 5. APRIORI
# --------------------------------------------------
st.header("5. Apriori")

min_support = st.slider(
    "Minimum Support",
    min_value=0.01,
    max_value=0.20,
    value=0.02,
    step=0.01
)

st.write(f"Current minimum support: {min_support:.2%}")
st.write(
    f"Approximately {round(len(baskets) * min_support)} baskets are required."
)

frequent_itemsets = apriori(
    basket_matrix,
    min_support=min_support,
    use_colnames=True,
    max_len=3
)

frequent_itemsets["size"] = frequent_itemsets["itemsets"].apply(len)

# Make itemsets easier to read
show_itemsets = frequent_itemsets.copy()
show_itemsets["itemsets"] = show_itemsets["itemsets"].apply(
    lambda x: ", ".join(sorted(x))
)

show_itemsets = show_itemsets.sort_values(
    "support",
    ascending=False
)

st.subheader("Frequent Itemsets")
st.dataframe(show_itemsets, width='stretch')

# --------------------------------------------------
# 6. TOP PAIRS
# --------------------------------------------------
st.header("6. Most Frequent Pairs")

pairs = frequent_itemsets[
    frequent_itemsets["size"] == 2
].nlargest(10, "support").copy()

if not pairs.empty:
    pairs["pair"] = pairs["itemsets"].apply(
        lambda x: " + ".join(sorted(x))
    )

    st.dataframe(
        pairs[["pair", "support"]],
        width='stretch'
    )

    chart_data = pairs[["pair", "support"]].set_index("pair")
    st.bar_chart(chart_data)
else:
    st.warning("No frequent pairs found. Try lowering min_support.")

# --------------------------------------------------
# 7. ASSOCIATION RULES
# --------------------------------------------------
st.header("7. Association Rules")

if not frequent_itemsets.empty:

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.1
    )

    if not rules.empty:

        rules["antecedents"] = rules["antecedents"].apply(
            lambda x: ", ".join(sorted(x))
        )

        rules["consequents"] = rules["consequents"].apply(
            lambda x: ", ".join(sorted(x))
        )

        rules["rule"] = (
            rules["antecedents"]
            + " → "
            + rules["consequents"]
        )

        rules_display = rules[
            ["rule", "support", "confidence", "lift"]
        ].sort_values(
            ["lift", "confidence"],
            ascending=False
        )

        st.dataframe(
            rules_display,
            width='stretch'
        )

        st.info(
            "Support = how frequent | "
            "Confidence = how often Y appears when X appears | "
            "Lift > 1 = positive association"
        )

    else:
        st.warning("No association rules found.")
else:
    st.warning("No frequent itemsets found.")

# --------------------------------------------------
# FINAL FLOW
# --------------------------------------------------
st.header("The Complete Flow")

st.success(
    "Raw Data → Clean Data → Baskets → Boolean Matrix → "
    "Apriori → Frequent Itemsets → Association Rules → Recommendations"
)
