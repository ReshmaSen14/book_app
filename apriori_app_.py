# ASSIGNMENT
# ASSOCIATION RULES - STREAMLIT CODE
# RESHMA SEN N

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

# Streamlit App Title
st.title(" Bookstore Association Rules")

# Sidebar for parameter selection
st.sidebar.header("Apriori Parameters")

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the dataset
    book = pd.read_csv(uploaded_file)
    
    # Ensure the dataset is binary (0s and 1s)
    if book.isin([0, 1]).all().all():
        st.success(" Data is correctly formatted as binary transactional data.")
    else:
        st.warning(" Warning: Non-binary values found! Check your dataset.")

    # Display first few rows
    st.write("### Preview of Uploaded Data")
    st.dataframe(book.head())

    # Convert data to boolean format for apriori
    df = book.astype(bool)

    # Sidebar sliders for Apriori parameters
    min_support = st.sidebar.slider("Minimum Support", 0.05, 0.3, 0.1, 0.01)
    min_confidence = st.sidebar.slider("Minimum Confidence", 0.3, 0.9, 0.5, 0.05)
    min_lift = st.sidebar.slider("Minimum Lift", 1.0, 2.0, 1.2, 0.1)

    # Button to generate rules
    if st.sidebar.button("Generate Rules"):
        # Apply Apriori Algorithm
        frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        rules = rules[rules['lift'] > min_lift]  # Apply Lift filter

        if not rules.empty:
            st.success(f" Generated {len(rules)} rules.")
            
            # Convert frozensets to strings for better readability
            rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x)))
            rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x)))

            # Display rules
            st.write("### Association Rules")
            st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

            # Visualizations
            st.write("### Rule Insights")

            # Scatter Plot: Support vs Confidence
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=rules, x='support', y='confidence', alpha=0.6, s=80, marker='o', ax=ax)
            plt.xlabel("Support")
            plt.ylabel("Confidence")
            plt.title("Support vs. Confidence")
            plt.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)

            # Scatter Plot: Lift vs Confidence
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=rules, x='confidence', y='lift', alpha=0.6, s=80, marker='o', ax=ax)
            plt.xlabel("Confidence")
            plt.ylabel("Lift")
            plt.title("Lift vs. Confidence")
            plt.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)

            # Barplot of Top 10 Rules by Lift
            st.write("### Top 10 Rules (by Lift)")
            top_rules = rules.nlargest(10, 'lift')
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=top_rules, x='lift', y='antecedents', hue='consequents', dodge=False, palette='viridis')
            plt.xlabel("Lift")
            plt.ylabel("Antecedents")
            plt.title("Top 10 Association Rules by Lift")
            plt.legend(title='Consequents', bbox_to_anchor=(1, 1))
            st.pyplot(fig)

        else:
            st.warning(" No rules found with the selected parameters. Try reducing thresholds.")
else:
    st.info(" Please upload a CSV file to proceed.")

