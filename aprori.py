import pandas as pd
import psycopg2
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

DB_CONFIG = {
    "host": "localhost",
    "database": "handbags",
    "user": "postgres",
    "password": "password",
    "port": "5432"
}

MIN_SUPPORT = 0.20
MIN_CONFIDENCE = 0.50

try:
    connection = psycopg2.connect(**DB_CONFIG)
    print("Database connection successful!")
except psycopg2.Error as e:
    print("Error connecting to PostgreSQL:")
    print(e)
    exit()

query = """
SELECT
    fs.Order_ID,
    fs.Product_ID,
    dp.Product_Name
FROM Fact_Sales fs
JOIN Dim_Product dp
    ON fs.Product_ID = dp.Product_ID
ORDER BY fs.Order_ID, fs.Product_ID;
"""

try:
    df = pd.read_sql_query(query, connection)
except Exception as e:
    print("Error reading data:")
    print(e)
    connection.close()
    exit()

print("\n============================================================")
print("RAW SALES DATA")
print("============================================================")
print(df.to_string(index=False))

if df.empty:
    print("\nNo sales data found in Fact_Sales.")
    connection.close()
    exit()

transactions = (
    df.groupby("order_id")["product_name"]
    .apply(lambda x: list(set(x)))
    .tolist()
)

print("\n============================================================")
print("TRANSACTIONS")
print("============================================================")

for i, transaction in enumerate(transactions, start=1):
    print(f"Transaction {i}: {transaction}")

print("\nTotal Transactions:", len(transactions))

encoder = TransactionEncoder()
encoded_array = encoder.fit(transactions).transform(transactions)

basket = pd.DataFrame(
    encoded_array,
    columns=encoder.columns_
)


print("\n============================================================")
print("RUNNING APRIORI ALGORITHM")
print("============================================================")

frequent_itemsets = apriori(
    basket,
    min_support=MIN_SUPPORT,
    use_colnames=True
)

if frequent_itemsets.empty:
    print("No frequent itemsets found.")
    print("Try lowering MIN_SUPPORT.")
    connection.close()
    exit()

frequent_itemsets["itemset_length"] = (
    frequent_itemsets["itemsets"].apply(len)
)

frequent_itemsets = frequent_itemsets.sort_values(
    by="support",
    ascending=False
).reset_index(drop=True)

print("\n============================================================")
print("FREQUENT ITEMSETS")
print("============================================================")

for index, row in frequent_itemsets.iterrows():
    items = ", ".join(sorted(row["itemsets"]))

    print(
        f"{index + 1}. "
        f"{{{items}}} "
        f"--> Support: {row['support']:.3f}"
    )

display_itemsets = frequent_itemsets.copy()

display_itemsets["itemsets"] = display_itemsets[
    "itemsets"
].apply(
    lambda x: ", ".join(sorted(x))
)

print("\n============================================================")
print("FREQUENT ITEMSETS TABLE")
print("============================================================")

print(
    display_itemsets[
        ["itemsets", "support", "itemset_length"]
    ].to_string(index=False)
)

print("\n============================================================")
print("GENERATING ASSOCIATION RULES")
print("============================================================")

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=MIN_CONFIDENCE
)

if rules.empty:
    print("\nNo association rules found.")
    print("Try lowering MIN_CONFIDENCE to 0.40 or 0.30.")
else:
    rules = rules[
        [
            "antecedents",
            "consequents",
            "antecedent support",
            "consequent support",
            "support",
            "confidence",
            "lift"
        ]
    ]

    rules = rules.sort_values(
        by="lift",
        ascending=False
    ).reset_index(drop=True)

    print("\n============================================================")
    print("ASSOCIATION RULES")
    print("============================================================")

    for index, row in rules.iterrows():
        antecedent = ", ".join(sorted(row["antecedents"]))
        consequent = ", ".join(sorted(row["consequents"]))

        print(f"\nRule {index + 1}")
        print("--------------------------------------------")
        print(f"IF customer buys: {antecedent}")
        print(f"THEN customer may buy: {consequent}")
        print(f"Support    : {row['support']:.3f}")
        print(f"Confidence : {row['confidence']:.3f}")
        print(f"Lift       : {row['lift']:.3f}")

    display_rules = rules.copy()

    display_rules["antecedents"] = display_rules[
        "antecedents"
    ].apply(
        lambda x: ", ".join(sorted(x))
    )

    display_rules["consequents"] = display_rules[
        "consequents"
    ].apply(
        lambda x: ", ".join(sorted(x))
    )

    print("\n============================================================")
    print("ASSOCIATION RULES TABLE")
    print("============================================================")

    print(
        display_rules[
            [
                "antecedents",
                "consequents",
                "support",
                "confidence",
                "lift"
            ]
        ].to_string(index=False)
    )

display_itemsets.to_csv(
    "frequent_itemsets.csv",
    index=False
)

print("\nFrequent itemsets saved to frequent_itemsets.csv")

if not rules.empty:
    display_rules.to_csv(
        "association_rules.csv",
        index=False
    )

    print("Association rules saved to association_rules.csv")

connection.close()

print("\n============================================================")
print("PROGRAM COMPLETED SUCCESSFULLY")
print("============================================================")