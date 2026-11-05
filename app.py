"""Run: python -m streamlit run app.py"""
from pathlib import Path
from html import escape
import json
import re
import pandas as pd
import streamlit as st
import recommender as engine

st.set_page_config(page_title='Basket & Rules | Abdelhafid Masmi',page_icon='🛒',layout='wide')
ROOT=Path(__file__).resolve().parent
st.markdown('''<style>
.stApp{background:#f5f8fd;color:#182d49}.block-container{max-width:1200px;padding-top:2rem}
header[data-testid="stHeader"]{background:#f5f8fd}
h1,h2,h3{color:#182d49;letter-spacing:-.5px}
[data-testid="stMarkdownContainer"] p,[data-testid="stWidgetLabel"] p,[data-testid="stCaptionContainer"] p{color:#314866}
[data-testid="stSidebar"]{background:#eaf0f9;border-right:1px solid #d8e2f0}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label{color:#203957}
div[data-baseweb="select"]>div,div[data-baseweb="input"]>div,div[data-baseweb="textarea"]{background:#fff!important;color:#182d49!important;border-color:#cbd8e9!important}
input,textarea{color:#182d49!important;-webkit-text-fill-color:#182d49!important;background:#fff!important}
div[data-baseweb="select"] span{color:#182d49}
[data-testid="stMetric"]{background:white;border:1px solid #dce5f1;border-radius:16px;padding:16px}
[data-testid="stMetricLabel"] p{color:#516784!important}[data-testid="stMetricValue"]{color:#164e9c;font-size:1.9rem}
.stButton button,.stDownloadButton button{border-radius:11px;min-height:44px;background:#fff;color:#174d9a;border:1px solid #bccfe9}
.stButton button p,.stDownloadButton button p{color:#174d9a!important;font-weight:650}
.stButton button[kind="primary"]{background:#235fcd!important;border-color:#235fcd!important}
.stButton button[kind="primary"] p{color:#fff!important}
.stButton button:hover,.stDownloadButton button:hover{border-color:#235fcd;box-shadow:0 4px 14px #235fcd17}
button:focus-visible{outline:3px solid #e9b64b!important;outline-offset:3px}
.hero{background:linear-gradient(115deg,#153861,#245fa0);padding:28px 32px;border-radius:23px;display:flex;gap:24px;align-items:center;flex-wrap:wrap;margin-bottom:18px}
.hero h1{color:#fff!important;margin:0;font-size:clamp(28px,3.5vw,42px);line-height:1.15}.hero p{color:#e1edff!important;margin:10px 0 0;font-size:16px}
.avatar{width:100px;height:100px;background:#2c74d9;border:3px solid #a2caff;border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-size:30px;font-weight:800;flex-shrink:0}
.eyebrow{color:#a9d3ff;font-size:12px;font-weight:700;letter-spacing:1.5px;margin-bottom:8px}
.author{font-size:14px!important}.pill{display:inline-block;background:#e8f0ff;color:#174d9a;border-radius:20px;padding:5px 11px;margin:4px 5px 4px 0;font-size:13px}
.product-title{font-size:24px;font-weight:750;color:#173b68;margin-bottom:12px}
.path{background:#fff;border:1px solid #dce5f1;border-radius:13px;padding:13px 17px;color:#4c6380;margin-bottom:20px;font-size:14px}
[data-baseweb="tab-list"]{gap:10px;border-bottom:1px solid #dce5f1}[data-baseweb="tab"] p{color:#24476f!important}
@media(max-width:600px){.hero{padding:20px}.avatar{width:76px;height:76px;font-size:25px}.block-container{padding-top:1rem}}
</style>''',unsafe_allow_html=True)

@st.cache_data
def read_inputs(rule_file,rule_stamp,data_file,data_stamp):
    table=engine.load_rules(rule_file)
    data=pd.read_csv(data_file,usecols=['transaction_id','item'])
    data['item']=data['item'].dropna().astype(str).str.strip().str.lower()
    catalogue=sorted(data['item'].dropna().unique())
    return table,catalogue,data['transaction_id'].nunique()

try:
    rule_file=ROOT/'rules.csv'
    if not rule_file.exists():rule_file=ROOT/'rules.json'
    data_file=ROOT/'grocery_transactions.csv'
    rules,catalogue,basket_count=read_inputs(str(rule_file),rule_file.stat().st_mtime_ns,str(data_file),data_file.stat().st_mtime_ns)
except (OSError,ValueError,KeyError,TypeError) as exc:
    st.error(f'Unable to load the homework data: {exc}');st.stop()
engine.rules=rules
# Preserve values across UI reruns. All basket mutations happen in callbacks.
if 'basket' not in st.session_state:st.session_state.basket=[]
if 'actions' not in st.session_state:st.session_state.actions=[]
def set_example(items):
    st.session_state.basket=items;st.session_state.actions=[]
    st.session_state.pop('unknown',None)
def add_products(items,explanation='Selected manually'):
    new=[x for x in items if x in catalogue and x not in st.session_state.basket]
    st.session_state.basket=list(st.session_state.basket)+new
    if new:st.session_state.actions.append({'added':new,'reason':explanation})
def remove_product(item):
    st.session_state.basket=[x for x in st.session_state.basket if x!=item]
def add_typed():
    items=list(dict.fromkeys(x.strip().lower() for x in re.split(r'[,;\n]+',st.session_state.typed) if x.strip()))
    st.session_state.unknown=[x for x in items if x not in catalogue]
    add_products(items)

with st.sidebar:
    st.markdown('### Your grocery assistant')
    st.caption('Association Rules · Homework')
    top_n=st.slider('Number of suggestions',1,8,3)
    st.markdown('**Try an example**')
    for label,items in [('🍝 Pasta meal',['pasta']),('☕ Breakfast',['coffee beans']),('🧴 Personal care',['shampoo'])]:
        st.button(label,on_click=set_example,args=(items,),width='stretch')
    st.button('Clear basket',on_click=set_example,args=([],),width='stretch')
    st.divider()
    st.caption('Select products, inspect the explanation, then add suggestions you like. Your results update automatically.')

st.markdown('''<div class="hero"><div class="avatar">AM</div><div><div class="eyebrow">VANIER COLLEGE · ASSOCIATION RULES</div><h1>Build a basket.<br>Discover what goes together.</h1><p>Explore recommendations and understand the rules behind them.</p><p class="author">Student: Abdelhafid Masmi</p></div></div><div class="path">① Choose your products &nbsp; → &nbsp; ② Explore matching rules &nbsp; → &nbsp; ③ Add suggestions and save your basket</div>''',unsafe_allow_html=True)
shop,summary,learn=st.tabs(['🛒 Basket & suggestions','🧾 My final basket','💡 Rules & learning'])
with shop:
    left,right=st.columns([.9,1.25],gap='large')
    with left:
        st.subheader('Your basket')
        st.multiselect('Products you already have',catalogue,key='basket',placeholder='Choose products…')
        with st.expander('Prefer typing? Add product names'):
            st.text_area('Separate names with commas or new lines',key='typed',placeholder='pasta, ground beef')
            st.button('Add typed products',on_click=add_typed,width='stretch')
            if st.session_state.get('unknown'):st.warning('Not in catalogue: '+', '.join(st.session_state.unknown))
        basket=list(st.session_state.basket)
        st.caption(f'{len(basket)} selected product(s). Use × in the selector to remove a product.')
        if basket:
            st.markdown(''.join('<span class="pill">'+escape(x.title())+'</span>' for x in basket),unsafe_allow_html=True)
            st.info('Your final basket and downloads are in the 🧾 My final basket tab.')
        else:st.info('Start with an example on the left, or choose a product above.')
    with right:
        st.subheader('Suggestions for you')
        results=engine.recommend(basket,top_n)
        if not basket:st.caption('Your matching recommendations will appear here.')
        elif not results:st.info('No rule matches this basket. Try adding another product or choose an example. The app does not invent recommendations.')
        else:
            st.caption(f'{len(results)} suggestion(s), ranked by lift. Existing basket products are excluded.')
            for r in results:
                # Find the strongest matching rule to expose its metrics alongside the original explanation.
                matches=rules[rules['antecedents'].map(lambda x:x.issubset(set(basket))) & rules['consequents'].map(lambda x:r['item'] in x)]
                best=matches.sort_values('lift',ascending=False,kind='stable').iloc[0]
                with st.container(border=True):
                    st.markdown('<div class="product-title">'+escape(r['item'].title())+'</div>',unsafe_allow_html=True)
                    a,b,c=st.columns(3)
                    a.metric('Lift',f"{best['lift']:.2f}×")
                    b.metric('Confidence',f"{best['confidence']:.0%}")
                    c.metric('Support',f"{best['support']:.1%}")
                    st.write(r['because'])
                    st.button('＋ Add to my basket',key='add_'+r['item'],on_click=add_products,args=([r['item']],r['because']),type='primary',width='stretch')
            st.download_button('Download these suggestions',json.dumps({'basket':basket,'recommendations':results},ensure_ascii=False,indent=2),'recommendations.json','application/json',width='stretch')
with summary:
    basket=list(st.session_state.basket)
    st.subheader('Your final basket')
    st.caption('Review your products, remove anything you do not want, and download the shopping list.')
    if not basket:st.info('Your basket is empty. Choose products in the first tab.')
    else:
        st.metric('Products in your basket',len(basket))
        for item in basket:
            a,b=st.columns([5,1])
            a.write('🛍️ **'+item.title()+'**')
            b.button('Remove',key='remove_'+item,on_click=remove_product,args=(item,))
        report={'student':'Abdelhafid Masmi','basket':basket,'accepted_actions':st.session_state.actions}
        st.download_button('Download shopping list (TXT)','MY SHOPPING LIST\n\n'+'\n'.join('☐ '+x.title() for x in basket),'shopping_list.txt','text/plain',width='stretch')
        st.download_button('Download full summary (JSON)',json.dumps(report,ensure_ascii=False,indent=2),'basket_summary.json','application/json',width='stretch')
        with st.expander('How recommendations helped'):
            st.caption('Historical additions are listed even if a product was subsequently removed.')
            for event in st.session_state.actions:st.write(', '.join(event['added'])+' — '+event['reason'])
            if not st.session_state.actions:st.write('No suggestions accepted yet.')
        st.caption('No checkout—your wallet is safe! This homework app creates a shopping list.')
with learn:
    a,b,c=st.columns(3)
    a.metric('Historical baskets',f'{basket_count:,}');b.metric('Products',len(catalogue));c.metric('Saved rules',len(rules))
    st.write('**Support:** how often both sides of the rule appear together in all baskets.')
    st.write('**Confidence:** among baskets with the IF products, how many also have the THEN product.')
    st.write('**Lift:** how much more frequent the THEN product is in IF baskets compared with its overall frequency. It is not a probability.')
    st.info('Rules describe historical associations. They do not guarantee a future purchase or prove causation. Apriori is run in the notebook; this app applies saved rules.')
    with st.expander('Explore the saved rules'):
        preview=rules.copy()
        for column in ['antecedents','consequents']:preview[column]=preview[column].map(lambda x:', '.join(sorted(x)))
        st.dataframe(preview,hide_index=True,width='stretch')
st.divider()
st.caption('Abdelhafid Masmi · Vanier College · Apriori learning lab')
