"""Run with: python -m streamlit run app.py"""
from pathlib import Path
import re
import json
import pandas as pd
import streamlit as st
import recommender as engine

st.set_page_config(page_title='Basket & Rules | Abdelhafid Masmi', page_icon='🛒', layout='wide')
st.markdown("""<style>
.stApp {background:#f7f9fd;color:#172b4d;}
.block-container {max-width:1120px;padding-top:2rem;}
h1,h2,h3 {color:#152d50;}
[data-testid="stMetric"] {background:white;border:1px solid #dce5f1;border-radius:14px;padding:16px;}
.intro {background:#173b68;color:white;border-radius:20px;padding:26px;margin-bottom:20px;}
.intro h1,.intro p {color:white !important;margin:0 0 8px;}
</style>""",unsafe_allow_html=True)
ROOT = Path(__file__).resolve().parent
st.markdown('<div class="intro"><h1>🛒 Basket & Rules</h1><p>Choose your products. Discover what goes together.</p><p>Student: Abdelhafid Masmi · Association Rules</p></div>',unsafe_allow_html=True)
try:
    rule_file = ROOT / 'rules.csv'
    if not rule_file.exists():
        rule_file = ROOT / 'rules.json'
    engine.rules = engine.load_rules(rule_file)
    data = pd.read_csv(ROOT / 'grocery_transactions.csv')
    catalogue = sorted(data['item'].dropna().astype(str).str.strip().str.lower().unique())
except (OSError, ValueError, KeyError, TypeError) as exc:
    st.error(f'Could not load the data: {exc}')
    st.stop()

shopping, learning = st.tabs(['Build a basket', 'Understand the rules'])
with shopping:
    st.subheader('1. Enter your basket')
    mode = st.radio('How would you like to enter products?', ['Choose from a list', 'Type my basket'], horizontal=True)
    if mode == 'Choose from a list':
        basket = st.multiselect('Products already in your basket', catalogue, key='chosen', placeholder='For example: pasta')
    else:
        raw = st.text_area('Type products separated by commas or new lines', placeholder='pasta, ground beef', key='typed')
        basket = sorted({item.strip().lower() for item in re.split(r'[,;\n]+', raw) if item.strip()})
        unknown = sorted(set(basket)-set(catalogue))
        if unknown:
            st.warning('Not found in the catalogue: '+', '.join(unknown)+'. Check spelling; these products cannot activate a rule.')
        st.caption('Spaces, repeated names, and uppercase letters are handled automatically.')
    how_many = st.slider('Maximum suggestions', 1, 8, 3)
    st.write('**Current basket:** '+(', '.join(basket) if basket else 'No products selected yet.'))
    # Only the recommendation function runs here; Apriori is not rerun.
    results = engine.recommend(basket, how_many)
    if not basket:
        st.info('Start with pasta, shampoo, or coffee beans to try the recommender.')
    if st.button('Get recommendations',type='primary',disabled=not basket):
        st.session_state.result = {'basket':list(basket),'how_many':how_many,'items':results}
    report = st.session_state.get('result')
    if report and report['basket'] == list(basket) and report['how_many'] == how_many:
        st.subheader('2. Your recommendations')
        if not report['items']:
            st.info('No saved rule matches this basket. Try another product or combination. No suggestions are invented.')
        else:
            st.success(f"Found {len(report['items'])} suggestion(s). Ranked by lift.")
            for r in report['items']:
                with st.container(border=True):
                    a,b = st.columns([4,1])
                    a.subheader(r['item'].title())
                    b.metric('Lift',f"{r['score']:.2f}×")
                    st.write(r['because'])
            st.subheader('3. Save your result')
            st.download_button('Download recommendations (JSON)',json.dumps(report,indent=2,ensure_ascii=False),file_name='basket_recommendations.json',mime='application/json')
    elif report:
        st.caption('Your basket or suggestion limit changed. Click Get recommendations to update the result.')
with learning:
    a,b = st.columns(2)
    a.metric('Products',len(catalogue));b.metric('Saved rules',len(engine.rules))
    st.write('**Workflow:** Transactions → baskets → True/False table → Apriori → filtered rules → saved file → recommendations.')
    st.write('**Support:** fraction of all baskets containing both sides of a rule.')
    st.write('**Confidence:** fraction of IF baskets that also contain THEN.')
    st.write('**Lift:** confidence divided by the suggested product’s overall frequency. It is not a probability.')
    st.write('A rule applies only when all IF products are in the basket. Existing products are excluded. If several rules suggest one product, the highest-lift rule is kept.')
    st.caption('Mining settings: support ≥ 0.02; maximum itemset length 3; lift ≥ 1.2; confidence ≥ 0.30; one consequent; supporting baskets ≥ 40.')
    preview = engine.rules.copy()
    for col in ['antecedents','consequents']:
        preview[col] = preview[col].map(lambda items:', '.join(sorted(items)))
    st.dataframe(preview,hide_index=True,width='stretch')
    st.info('Patterns describe past co-purchases. They do not prove causation or guarantee what a customer wants. Rare or unfamiliar products may have no matching rule.')
