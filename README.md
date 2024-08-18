# Product Recommendation System

This project is a product recommendation system that utilizes clustering and cosine similarity to suggest items to users based on their ratings. The project is implemented in Python using libraries such as NumPy, pandas, and scikit-learn.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Features](#features)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Overview

The product recommendation system processes user reviews and ratings to provide personalized product recommendations. It uses K-means clustering and cosine similarity to analyze the data and generate recommendations.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/SwatiMane92/product-recommendation-system.git
   cd product-recommendation-system
   ```

2. Install the required packages:

   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```

3. Ensure the `Reviews.csv` dataset is in the project directory.

## Dataset

The dataset used for this project is `Reviews.csv`, which contains user reviews and ratings of various products. The relevant columns used in this project are:

- `Id`: The unique identifier of the user.
- `ProductId`: The unique identifier of the product.
- `Score`: The rating given by the user to the product.

## Features

- **Data Preprocessing**: Handles missing values and duplicates in the dataset.
- **Cosine Similarity**: Computes the similarity between user-item interactions to recommend products.
- **K-Means Clustering**: Groups similar users together to enhance recommendation accuracy.
- **Top-k Recommendations**: Generates top-k product recommendations for a given user.

## Usage

1. **Data Preprocessing**:

   ```python
   df = pd.read_csv('Reviews.csv')
   df = df.dropna()
   df1 = df.iloc[:10000, :]
   ```

2. **Cosine Similarity-based Recommendations**:

   ```python
   pivot_table = ratings_df.pivot_table(index='Id', columns='ProductId', values='Score', fill_value=0)
   items_similarity = cosine_similarity(pivot_table)
   
   # Example: Get top-k recommendations for a given user
   user_id = 4
   k = 5
   ```

3. **K-Means Clustering for Recommendations**:

   ```python
   num_clusters = 5
   kmeans = KMeans(n_clusters=num_clusters, random_state=42)
   cluster_labels = kmeans.fit_predict(pivot_table)
   
   # Example: Get top-k recommendations
   user_id = 5
   ```

4. **Recommendation Function**:

   ```python
   def recommend_items(ratings_df):
       filtered_recommendations = ratings_df[ratings_df['Score'] >= 4].head(5)
       return filtered_recommendations
   ```

5. **Running the Code**: Execute the code in a Jupyter Notebook or Python environment to see the output.

## Results

- **Cosine Similarity**: Provides recommendations based on user-item interaction similarity.
- **K-Means Clustering**: Groups users into clusters to recommend products based on average ratings within the cluster.
- **Top-k Recommendations**: Displays the top-k product recommendations for a specific user.

