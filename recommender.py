"""Rule matching and ranking based on the Part 3 notebook."""
from pathlib import Path
import pandas as pd

rules = None

def load_rules(path):
    table = pd.read_json(path) if Path(path).suffix == '.json' else pd.read_csv(path)
    required = {'antecedents', 'consequents', 'support', 'confidence', 'lift', 'n_baskets'}
    if not required.issubset(table.columns):
        raise ValueError('The rule file is missing required columns.')
    for column in ['antecedents', 'consequents']:
        # Part 3 exports comma-separated text; JSON list exports are also accepted.
        table[column] = table[column].apply(lambda value: frozenset(
            str(item).strip().lower() for item in
            (value.split(',') if isinstance(value, str) else value) if str(item).strip()
        ))
    if not table['consequents'].map(len).eq(1).all():
        raise ValueError('This recommender requires one consequent per rule.')
    for column in ['support', 'confidence', 'lift', 'n_baskets']:
        table[column] = pd.to_numeric(table[column], errors='raise')
    if not table[['support','confidence','lift','n_baskets']].notna().all().all():
        raise ValueError('Rule scores contain missing values.')
    return table

def recommend(basket, how_many=3):
    # STEP 7.1: CONVERT THE BASKET INTO A SET
    '''
    Example: 
    ['pasta', 'ground beef', 'pasta']
    becomes:
    ['pasta', 'ground beef']
    Sets remove duplicated and make membership checks convenient
    '''
    
    basket = {str(item).strip().lower() for item in basket if str(item).strip()} # a set makes the "is it inside?" test easy
    
    # STEP 7.2: CREATE DICTIONARIES TO STORE RECOMMENDATIONS
    '''
    Dictionary storing each recommended product and its best lift score.add()
    {
        'tomato sauce': 2.4,
        'parmesan': 1.8
    }
    
    '''
    suggestions = {} # product  -> its best score so far
    
    '''
    Dictionary storing the explanation associated with each product.
    
    The product name is the key, and its explanation is the value
    
    Example:
    
    {
        'tomato sauce': 'bacause you bought pasta ...'
    }
    '''
    reason = {}
    
    # STEP 7.3 EXAMINE EVERY ASSOCIATION RULE
    '''
    `rules` is a pandas dataFrame containing the assicaiation rules learned in the previous steps
    `.iterrows()` processes the DataFrame on row at a time.
    `_` receives the row index, which we do not need.
    `rule` contains the current row's data.
    '''
    # Look at every rule we kep in STEP 5
    for _, rule in rules.iterrows():
        # STEP 3A: GET THE ANTECEDENT (THE "IF" PART)
        '''
        Retrieve the products required for this rule to apply.
        Example:
        {'pasta', 'ground beef'}
        
        Convert them into a set for easy comparison with the basket
        '''
        
        if_part = set(rule['antecedents'])
        
        
        # STEP 3B: GET THE CONSEQUENT (THE "THEN" PART)
        '''
        Retrieve the product suggested by the rule.
        `list()` converts the consequent collection into a list
        `[0]` selects the first product
        
        '''
        then_part = list(rule['consequents'])[0]
        # print(if_part, then_part)
        
        # SETP 7.4 CHECK WHETHER THE RULE APPLIES
        '''
        `.issubset(basket)` checks whether EVERY product in `if_part` is present in the customer's basket
        `not` reverses the result
        It even one required product is missing, skip this rule
        '''
        
        if not if_part.issubset(basket):
            continue
        
        # STEP 7.5 DO NOT RECOMMEND PRODCUTS ALREADY IN THE BASKET
        '''
        Check whether the suggested product is already in the basket. 
        Never recommend something the customer already has
        If it is, skip this rule and examine the next one.
        '''
        
        if then_part in basket:
            continue
        
        # STEP 7.6 KEPP THE STRONGEST REULE FOR EACH PRODUCT
        
        '''
        A product may be suggested by multiply rules.
        For example:
        
        pasta                   -> tomato sauce lift = 1.7
        pasta + ground beef     -> tomato sauce lift = 2.4
        
        Keep the rule with the highest lift score
        `or` means the condition is true if the product is new OR 
        the current rule has a higher lift that the previous stored score.
         
        '''
        # if (then_part not in suggestions
        #     or rule['lift'] > suggestions[then_part]
        #     ):
        if then_part not in suggestions or rule['lift'] > suggestions[then_part]:
            
            # Store the strongest lift score found for this product
            suggestions[then_part] = rule['lift']
            '''
            Sort the antecedent products alphabetically and join them into a readable phrase
            `' and '.join(if_part)`
            Example:
            
            {'pasta', 'ground beef'}
            becomes:
            'ground beef and pasta'
            
            '''
            trigger = ' and '.join(sorted(if_part))
            
            # Build an explanation using f-string.
            '''
            {trigger} inserts the products that activated the rule.
            {then_part} inserts the recommended product
            
            {rule['confidence']:.0%} formats confidence as a percentage with zero decimal places.
            
            '''
            
            reason[then_part] = (
                f"because you bought {trigger} - "
                f"{rule['confidence']:.0%} of baskets with "
                f"{trigger} also contain {then_part}, "
                f"{rule['lift']:.1f}x more often than average "
                f"(based on {int(rule['n_baskets'])} observed baskets)"
            )
            
            # Save the explanation under the recommended product's name.
            # if a stronger rule is found later, this explanation will be replaced along with the score.
    # STEP 7.7 SORT THE RECOMMENDATIONS
    
    # Sort the suggestions from the highest score to the lowest, keep the top ones.
    # sorted(..., key =..., reverse=True) sort by the score, biggest first
    best = sorted(
        suggestions.items(), 
        key=lambda pair: pair[1], 
        reverse=True
        )[:how_many]
    
    '''
    
    Example result:
    
    [
        {
            'item': 'tomato sauce,
            'score': 2.4,
            'because': 'because you bought pasta ..'
        }
    ]
    if no eligible recommendations exist, this returns an empty list
    '''
    
    return [
        {
            'item': item,
            'score': round(score, 2),
            'because': reason[item]
        } for item, score in best
    ]