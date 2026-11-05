# Association Rule Recommender
Student: Abdelhafid Masmi

## Run
Activate vanier-dmp, open this folder, then:
```
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Use the selector or type comma/newline-separated products. Try pasta, pasta + ground beef, beer + tortilla chips, diapers, yogurt + berries, or shampoo.

## Files
- Apriori-Part-3-masmi.ipynb: original notebook order, loading TODO completed; run all cells to regenerate rules.
- recommender.py: Part 3 matching, highest lift per product, top-N, and explanations.
- app.py: list/text input, validation, displayed recommendations and download.
- rules.csv / rules.json: saved rules from the same 2,472-basket dataset.
- grocery_transactions.csv: 50-product catalogue and mining input.

The app prefers rules.csv and falls back to rules.json if CSV is absent. Both are equivalent exports, not combined. Keep them together with the app.

## Adaptations from Part 3
The notebook loading TODO assigns to rules (the DataFrame used by the function), not rule (the loop's current row). CSV text is parsed back into sets. The app normalizes user input, sorts the explanation's trigger names for readability, and uses 'observed baskets' rather than asserting the data is real. The professor's matching and lift-only ranking method is retained. Notebook demonstration cells remain intact. No Streamlit app is launched inside Jupyter.

This is a learning project. Function checks do not measure recommendation accuracy on unseen customers. The app does not place orders.
