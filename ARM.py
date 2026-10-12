# ---------------------------------------------------
# Implementation of Association Rule Mining
# Apriori Algorithm - Supermarket Dataset
# ---------------------------------------------------

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ---------------------------------------------------
# 1. Create Supermarket Transaction Data
# ---------------------------------------------------

transactions = [
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread'],
    ['Milk', 'Bread', 'Butter', 'Eggs'],
    ['Bread', 'Eggs'],
    ['Milk', 'Bread', 'Eggs'],
    ['Milk', 'Butter'],
    ['Bread', 'Butter', 'Eggs'],
    ['Milk', 'Bread', 'Butter'],
    ['Milk', 'Bread', 'Butter', 'Eggs']
]

print("Supermarket Transactions:")
for i, transaction in enumerate(transactions, start=1):
    print(f"Transaction {i}: {transaction}")


# ---------------------------------------------------
# 2. Convert transactions into one-hot encoded format
# ---------------------------------------------------

te = TransactionEncoder()

te_array = te.fit(transactions).transform(transactions)

df = pd.DataFrame(
    te_array,
    columns=te.columns_
)

print("\nOne-Hot Encoded Transaction Data:")
print(df)


# ---------------------------------------------------
# 3. Generate Frequent Itemsets using Apriori
# ---------------------------------------------------

frequent_itemsets = apriori(
    df,
    min_support=0.3,
    use_colnames=True
)

print("\nFrequent Itemsets:")
print(frequent_itemsets)


# ---------------------------------------------------
# 4. Generate Association Rules
# ---------------------------------------------------

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.6
)

print("\nAssociation Rules:")

# Display important columns
print(
    rules[
        ['antecedents',
         'consequents',
         'support',
         'confidence',
         'lift']
    ]
)


# ---------------------------------------------------
# 5. Display rules sorted by Lift
# ---------------------------------------------------

rules = rules.sort_values(
    by='lift',
    ascending=False
)

print("\nAssociation Rules Sorted by Lift:")

print(
    rules[
        ['antecedents',
         'consequents',
         'support',
         'confidence',
         'lift']
    ]
)