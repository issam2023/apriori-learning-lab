import streamlit as st
from textwrap import dedent
from textwrap import dedent
import pandas as pd
from datetime import date
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ============================================================
# PAGE
# ============================================================
st.set_page_config(
    page_title="Apriori Learning Lab",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS
# ============================================================
st.markdown(dedent("""
<style>

/* Main page */
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 3rem;
    max-width: 1550px;
}

/* Header */
.hero {
    background: linear-gradient(110deg, #063b72 0%, #075da8 55%, #087ac4 100%);
    padding: 25px 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,.18);
}

.hero-grid {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 25px;
    flex-wrap: wrap;
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    margin: 0;
}

.hero-subtitle {
    font-size: 18px;
    margin-top: 5px;
    opacity: .92;
}

.author-box {
    background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.28);
    padding: 13px 18px;
    border-radius: 13px;
    min-width: 250px;
}

/* Intro box */
.intro {
    border: 1px solid #2789d8;
    border-radius: 16px;
    padding: 18px 22px;
    margin-bottom: 20px;
    background: rgba(30,120,200,.08);
}

/* Flow */
.flow {
    display: flex;
    align-items: center;
    gap: 7px;
    margin: 18px 0 25px 0;
    flex-wrap: wrap;
}

.flow-box {
    padding: 11px 14px;
    border-radius: 11px;
    font-weight: 700;
    text-align: center;
    border: 1px solid rgba(255,255,255,.12);
}

.flow-arrow {
    font-size: 23px;
    font-weight: bold;
}

.blue { background: rgba(40,140,240,.20); }
.green { background: rgba(35,190,100,.20); }
.yellow { background: rgba(240,190,40,.20); }
.purple { background: rgba(140,80,220,.20); }
.red { background: rgba(235,70,90,.20); }

/* Step cards */
.step-card {
    border-radius: 15px;
    padding: 15px 18px;
    margin: 8px 0 14px 0;
    border: 1px solid rgba(120,150,190,.35);
    background: rgba(100,130,160,.07);
}

.step-number {
    display: inline-block;
    background: #0877c9;
    color: white;
    border-radius: 50%;
    width: 34px;
    height: 34px;
    line-height: 34px;
    text-align: center;
    font-weight: bold;
    margin-right: 8px;
}

.step-title {
    font-size: 24px;
    font-weight: 750;
}

/* Metrics */
.metric-card {
    border-radius: 14px;
    padding: 15px;
    text-align: center;
    border: 1px solid rgba(100,150,200,.3);
    background: rgba(40,120,200,.10);
}

.metric-big {
    font-size: 27px;
    font-weight: 800;
}

.metric-label {
    opacity: .75;
}

/* Concept boxes */
.concept {
    border-left: 5px solid #1687d9;
    padding: 12px 16px;
    margin: 12px 0;
    border-radius: 8px;
    background: rgba(20,130,210,.10);
}

/* Footer */
.footer {
    margin-top: 40px;
    padding: 18px;
    text-align: center;
    border-top: 1px solid rgba(150,150,150,.25);
    opacity: .8;
    font-size: 14px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(100,150,200,.25);
}

</style>
"""), unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    return pd.read_csv("grocery_transactions.csv")


df = load_data()

clean_df = df.copy()
clean_df["item"] = (
    clean_df["item"]
    .astype(str)
    .str.strip()
    .str.lower()
)

basket_series = (
    clean_df.groupby("transaction_id")["item"]
    .apply(lambda items: sorted(set(items)))
)

basket_ids = basket_series.index.tolist()
baskets = basket_series.tolist()

basket_df = pd.DataFrame({
    "transaction_id": basket_ids,
    "basket": baskets
})

te = TransactionEncoder()
encoded = te.fit(baskets).transform(baskets)

basket_matrix = pd.DataFrame(
    encoded,
    columns=te.columns_,
    index=basket_ids
)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🛒 Apriori Lab")
    st.caption("Interactive Learning Application")

    st.divider()

    page = st.radio(
        "Learning Steps",
        [
            "🏠 Overview",
            "1️⃣ Raw Data",
            "2️⃣ Clean Data",
            "3️⃣ Baskets",
            "4️⃣ Boolean Matrix",
            "5️⃣ Apriori",
            "6️⃣ Frequent Itemsets",
            "7️⃣ Association Rules",
            "8️⃣ Recommendations"
        ]
    )

    st.divider()

    st.caption("Association Rule Mining")
    st.caption("Apriori Learning Lab")


# ============================================================
# HEADER
# ============================================================
st.title("🛒 Association Rule Mining — Apriori Learning Lab")
st.subheader("Interactive step-by-step learning using real supermarket data")

c1, c2 = st.columns([3, 1])

with c1:
    st.info(
        "Follow supermarket data from raw transactions "
        "to association rules and recommendations."
    )

with c2:
    st.markdown("""
**Abdelhafid Masmi**  
Vanier College — DMP 2026  
September 25, 2026
""")

# ============================================================
# LEARNING FLOW
# ============================================================
st.markdown("### 🔄 Complete Learning Flow")

st.image(
    "apriori_workflow.png",
    use_container_width=True
)

# ============================================================
# SUPPORT CONTROL - USED BY MULTIPLE PAGES
# ============================================================
with st.sidebar:
    st.divider()
    st.markdown("### ⚙️ Apriori Settings")

    min_support = st.slider(
        "Minimum support",
        min_value=0.01,
        max_value=0.15,
        value=0.02,
        step=0.01
    )

frequent_itemsets = apriori(
    basket_matrix,
    min_support=min_support,
    use_colnames=True,
    max_len=3
)

frequent_itemsets["size"] = (
    frequent_itemsets["itemsets"].apply(len)
)


# ============================================================
# RULES
# ============================================================
rules = pd.DataFrame()

if not frequent_itemsets.empty:
    try:
        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=0.10
        )
    except Exception:
        rules = pd.DataFrame()


# ============================================================
# OVERVIEW
# ============================================================
if page == "🏠 Overview":

    st.markdown(dedent("""
    <div class="intro">
        <h2>📊 Overview</h2>
        This application demonstrates <b>Association Rule Mining</b>
        using the <b>Apriori algorithm</b>.
        Follow the data from individual supermarket rows all the way
        to frequent patterns, association rules, and recommendations.
    </div>
    """), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("CSV Rows", f"{len(df):,}")

    with c2:
        st.metric(
            "Transactions",
            f"{df['transaction_id'].nunique():,}"
        )

    with c3:
        st.metric(
            "Unique Products",
            f"{clean_df['item'].nunique():,}"
        )

    with c4:
        st.metric(
            "Frequent Itemsets",
            f"{len(frequent_itemsets):,}"
        )

    st.subheader("How the data changes")

    a, b, c, d = st.columns(4)

    with a:
        st.markdown("### 1️⃣ Raw Data")
        st.caption("One product per row")
        st.dataframe(
            df[["transaction_id", "item"]].head(7),
            width="stretch",
            hide_index=True
        )

    with b:
        st.markdown("### 2️⃣ Clean Data")
        st.caption("Lowercase + remove spaces")
        st.dataframe(
            clean_df[["transaction_id", "item"]].head(7),
            width="stretch",
            hide_index=True
        )

    with c:
        st.markdown("### 3️⃣ Baskets")
        st.caption("Group products by transaction")
        st.dataframe(
            basket_df.head(5),
            width="stretch",
            hide_index=True
        )

    with d:
        st.markdown("### 4️⃣ Boolean Matrix")
        st.caption("1 = present, 0 = absent")
        overview_matrix = (
            basket_matrix.head(5)
            .astype(int)
        )
        st.dataframe(
            overview_matrix,
            width="stretch"
        )

    st.info(
        "Main idea: we transform raw transaction rows into a Boolean "
        "matrix so Apriori can search for products that frequently occur together."
    )


# ============================================================
# RAW DATA
# ============================================================
elif page == "1️⃣ Raw Data":

    st.markdown(
        '<span class="step-number">1</span>'
        '<span class="step-title">Raw Data</span>',
        unsafe_allow_html=True
    )

    st.write(
        "The CSV is in **long format**. "
        "Each row represents one product inside a transaction."
    )

    c1, c2 = st.columns(2)

    c1.metric("Rows", f"{len(df):,}")
    c2.metric(
        "Transactions",
        f"{df['transaction_id'].nunique():,}"
    )

    st.dataframe(
        df.head(50),
        width="stretch"
    )

    st.markdown(dedent("""
    <div class="concept">
    <b>Concept:</b> Several rows can belong to the same transaction.
    We must group those rows before Apriori can understand which
    products were purchased together.
    </div>
    """), unsafe_allow_html=True)


# ============================================================
# CLEAN
# ============================================================
elif page == "2️⃣ Clean Data":

    st.markdown(
        '<span class="step-number">2</span>'
        '<span class="step-title">Clean the Data</span>',
        unsafe_allow_html=True
    )

    st.write(
        "We remove extra spaces and convert product names to lowercase."
    )

    st.code(
        'df["item"] = df["item"].str.strip().str.lower()',
        language="python"
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Before")
        st.dataframe(
            df[["transaction_id", "item"]].head(20),
            width="stretch",
            hide_index=True
        )

    with right:
        st.subheader("After")
        st.dataframe(
            clean_df[["transaction_id", "item"]].head(20),
            width="stretch",
            hide_index=True
        )

    st.success(
        "Cleaning prevents names such as Milk, MILK and ' milk ' "
        "from being treated as different products."
    )


# ============================================================
# BASKETS
# ============================================================
elif page == "3️⃣ Baskets":

    st.markdown(
        '<span class="step-number">3</span>'
        '<span class="step-title">Create Baskets</span>',
        unsafe_allow_html=True
    )

    st.write(
        "Rows with the same `transaction_id` are grouped together."
    )

    transaction = st.selectbox(
        "Select a transaction to follow",
        basket_ids
    )

    original = clean_df[
        clean_df["transaction_id"] == transaction
    ][["transaction_id", "item"]]

    basket = basket_series.loc[transaction]

    left, middle, right = st.columns([2, .5, 2])

    with left:
        st.subheader("Rows")
        st.dataframe(
            original,
            width="stretch",
            hide_index=True
        )

    with middle:
        st.markdown(
            "<h1 style='text-align:center;padding-top:60px;'>→</h1>",
            unsafe_allow_html=True
        )

    with right:
        st.subheader("Basket")
        st.write(f"**{transaction}**")
        for item in basket:
            st.write(f"🛒 {item}")

    st.success(
        f"{len(original)} rows became one basket containing "
        f"{len(basket)} unique products."
    )


# ============================================================
# BOOLEAN MATRIX
# ============================================================
elif page == "4️⃣ Boolean Matrix":

    st.markdown(
        '<span class="step-number">4</span>'
        '<span class="step-title">Boolean Matrix — TransactionEncoder</span>',
        unsafe_allow_html=True
    )

    st.write(
        "`TransactionEncoder` converts baskets into True/False values."
    )

    transaction = st.selectbox(
        "Select a transaction",
        basket_ids
    )

    st.subheader("Basket")

    selected_basket = basket_series.loc[transaction]

    st.write(selected_basket)

    st.markdown("### ↓ TransactionEncoder ↓")

    selected_matrix = (
        basket_matrix.loc[[transaction]]
        .astype(int)
    )

    # Only show columns with products present for easier learning
    present_columns = [
        col for col in selected_matrix.columns
        if selected_matrix.iloc[0][col] == 1
    ]

    st.write("#### Products present in this transaction")

    st.dataframe(
        selected_matrix[present_columns],
        width="stretch"
    )

    with st.expander("Show complete Boolean matrix"):
        st.dataframe(
            basket_matrix.head(20).astype(int),
            width="stretch"
        )

    st.info(
        "1 / True = product is in the basket. "
        "0 / False = product is not in the basket."
    )


# ============================================================
# APRIORI
# ============================================================
elif page == "5️⃣ Apriori":

    st.markdown(
        '<span class="step-number">5</span>'
        '<span class="step-title">Apriori</span>',
        unsafe_allow_html=True
    )

    st.write(
        "Apriori searches the Boolean matrix for item combinations "
        "that occur frequently enough."
    )

    required = round(len(baskets) * min_support)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Minimum Support",
        f"{min_support:.0%}"
    )

    c2.metric(
        "Approx. baskets required",
        f"{required:,}"
    )

    c3.metric(
        "Frequent Itemsets",
        f"{len(frequent_itemsets):,}"
    )

    st.code(
        f"""frequent_itemsets = apriori(
    basket_matrix,
    min_support={min_support},
    use_colnames=True,
    max_len=3
)""",
        language="python"
    )

    st.markdown(dedent("""
    <div class="concept">
    <b>Support</b> tells us how frequently an itemset appears
    in all transactions.
    </div>
    """), unsafe_allow_html=True)

    st.latex(
        r"Support(X)=\frac{Transactions\ containing\ X}"
        r"{Total\ transactions}"
    )


# ============================================================
# FREQUENT ITEMSETS
# ============================================================
elif page == "6️⃣ Frequent Itemsets":

    st.markdown(
        '<span class="step-number">6</span>'
        '<span class="step-title">Frequent Itemsets</span>',
        unsafe_allow_html=True
    )

    display_frequent = frequent_itemsets.copy()

    display_frequent["itemset"] = (
        display_frequent["itemsets"]
        .apply(lambda x: " + ".join(sorted(x)))
    )

    display_frequent = display_frequent[
        ["itemset", "support", "size"]
    ].sort_values(
        "support",
        ascending=False
    )

    st.dataframe(
        display_frequent,
        width="stretch",
        hide_index=True
    )

    pairs = display_frequent[
        display_frequent["size"] == 2
    ].head(12)

    if not pairs.empty:
        st.subheader("Top Frequent Pairs")

        chart = pairs[
            ["itemset", "support"]
        ].set_index("itemset")

        st.bar_chart(chart)

    st.info(
        "A frequent itemset is a group of products whose support "
        "is greater than or equal to the minimum support."
    )


# ============================================================
# RULES
# ============================================================
elif page == "7️⃣ Association Rules":

    st.markdown(
        '<span class="step-number">7</span>'
        '<span class="step-title">Association Rules</span>',
        unsafe_allow_html=True
    )

    if rules.empty:
        st.warning(
            "No rules found. Try lowering minimum support."
        )

    else:
        rules_display = rules.copy()

        rules_display["antecedents_text"] = (
            rules_display["antecedents"]
            .apply(lambda x: ", ".join(sorted(x)))
        )

        rules_display["consequents_text"] = (
            rules_display["consequents"]
            .apply(lambda x: ", ".join(sorted(x)))
        )

        rules_display["rule"] = (
            rules_display["antecedents_text"]
            + " → "
            + rules_display["consequents_text"]
        )

        rules_display = rules_display[
            ["rule", "support", "confidence", "lift"]
        ].sort_values(
            ["lift", "confidence"],
            ascending=False
        )

        st.dataframe(
            rules_display,
            width="stretch",
            hide_index=True
        )

        st.markdown(dedent("""
        <div class="concept">
        <b>Support</b> = how frequent the combination is.<br>
        <b>Confidence</b> = when X occurs, how often Y occurs.<br>
        <b>Lift</b> = how much stronger the relationship is
        compared with normal chance.
        </div>
        """), unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        c1.metric("Support", "Frequency")
        c2.metric("Confidence", "X → Y")
        c3.metric("Lift > 1", "Positive association")


# ============================================================
# RECOMMENDATIONS
# ============================================================
elif page == "8️⃣ Recommendations":

    st.markdown(
        '<span class="step-number">8</span>'
        '<span class="step-title">Product Recommendations</span>',
        unsafe_allow_html=True
    )

    st.write(
        "Choose a product. The app searches the discovered rules "
        "for possible recommendations."
    )

    products = sorted(clean_df["item"].unique())

    selected_product = st.selectbox(
        "🛒 Customer has:",
        products
    )

    if rules.empty:
        st.warning(
            "No rules available. Lower minimum support."
        )

    else:
        matching = rules[
            rules["antecedents"].apply(
                lambda x: selected_product in x
            )
        ].copy()

        if matching.empty:
            st.warning(
                "No recommendation found for this product "
                "with the current support threshold."
            )

        else:
            matching = matching.sort_values(
                ["lift", "confidence"],
                ascending=False
            )

            recommendations = []

            for _, row in matching.iterrows():
                for product in row["consequents"]:
                    if product != selected_product:
                        recommendations.append({
                            "Recommended Product": product,
                            "Confidence": row["confidence"],
                            "Lift": row["lift"],
                            "Support": row["support"]
                        })

            rec_df = pd.DataFrame(recommendations)

            if not rec_df.empty:
                rec_df = (
                    rec_df
                    .sort_values(
                        ["Lift", "Confidence"],
                        ascending=False
                    )
                    .drop_duplicates(
                        subset=["Recommended Product"]
                    )
                    .head(10)
                )

                st.success(
                    f"Recommendations related to: {selected_product}"
                )

                st.dataframe(
                    rec_df,
                    width="stretch",
                    hide_index=True
                )

                best = rec_df.iloc[0]

                st.markdown(
                    f"""
                    ### 🛒 Customer has
                    **{selected_product}**

                    ### ↓ Recommendation

                    ## 💡 {best['Recommended Product']}

                    **Confidence:** {best['Confidence']:.2%}  
                    **Lift:** {best['Lift']:.2f}  
                    **Support:** {best['Support']:.2%}
                    """
                )


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    f"""
    <div class="footer">
        <b>Association Rule Mining — Apriori Learning Lab</b><br>
        Vanier College • DMP 2026<br><br>
        Dataset: grocery_transactions.csv •
        Rows: {len(df):,} •
        Transactions: {df['transaction_id'].nunique():,} •
        September 25, 2026
    </div>
    """,
    unsafe_allow_html=True
)
