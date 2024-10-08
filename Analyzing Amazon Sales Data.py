import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
import time
# Set up logging for retraining
logging.basicConfig(filename='retrain.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def retrain_model():
    logging.info('Starting model retraining.')

    # Load the dataset
    df = pd.read_csv('amazon.csv')


    # Displays the first few rows of the dataset
    print(df.head())

# Displays the last few rows of the dataset
    print(df.tail()) 

# Finds shape of the dataset
    print(df.shape)
    print("Number of Rows:", df.shape[0])
    print("Number of Columns:", df.shape[1])

    # Displays info of the dataset
    print(df.info())

    # Check for duplicated
    data_dup = df.duplicated().any()
    print("Are there duplicates?", data_dup)

    # Drop duplicates
    df = df.drop_duplicates()
    print("Shape after removing duplicates:", df.shape)

    # Displays summary statistics
    print(df.describe())

    # Check for missing values
    print(df.isnull().sum())

    # Check the data types of the columns
    print(df.dtypes)

    # Convert 'discounted_price' and 'actual_price' to numeric, handling any errors
    df['discounted_price'] = pd.to_numeric(
        df['discounted_price'].replace('₹', '', regex=True).str.replace(',', '').str.strip(), 
        errors='coerce'
    )
    df['actual_price'] = pd.to_numeric(
        df['actual_price'].replace('₹', '', regex=True).str.replace(',', '').str.strip(), 
        errors='coerce'
    )
    df['discount_percentage'] = pd.to_numeric(
        df['discount_percentage'].str.replace('%', '').str.strip(), 
        errors='coerce'
    )
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['rating_count'] = pd.to_numeric(df['rating_count'].str.replace(',', ''), errors='coerce')

    # Fill missing values
    df.ffill(inplace=True)

    # Visualizations for Numeric Features
    numeric_features = ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count']
    for feature in numeric_features:
        plt.figure(figsize=(10, 6))
        df[feature].hist(bins=30)
        plt.title(f'Histogram of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.show()

    # Simulate Orders Dataset
    orders_df = df[['product_id', 'discounted_price']].copy()
    orders_df['Order ID'] = ['ORD' + str(i) for i in range(1, len(orders_df) + 1)]

    # Simulate Date Column
    np.random.seed(0)  # For reproducibility
    dates = pd.date_range(start='2023-01-01', periods=len(orders_df), freq='D')
    orders_df['date'] = np.random.choice(dates, size=len(orders_df))

    # Saving the simulated orders dataset to a CSV file
    orders_df.to_csv('orders.csv', index=False)

    # Saving the product dataset
    products_df = df[['product_id', 'product_name', 'category', 'discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count']].copy()
    products_df.to_csv('products.csv', index=False)

    # Simulate Customer Dataset
    customers_df = pd.DataFrame({
        'customer_id': ['CUST' + str(i) for i in range(1, len(orders_df) + 1)],
        'location': np.random.choice(['North America', 'Europe', 'Asia', 'South America', 'Australia'], size=len(orders_df))
    })

    # Saving the simulated customers dataset to a CSV file
    customers_df.to_csv('customers.csv', index=False)

    # Load and Preview the simulated datasets
    orders_df_loaded = pd.read_csv('orders.csv')
    products_df_loaded = pd.read_csv('products.csv')
    customers_df_loaded = pd.read_csv('customers.csv')

    # Bar Chart for Customer Locations
    if 'location' in customers_df_loaded.columns:
        plt.figure(figsize=(10, 6))
        customers_df_loaded['location'].value_counts().plot(kind='bar')
        plt.title('Distribution of Customer Locations')
        plt.xlabel('Location')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()
    else:
        print("Column 'location' not found in the dataset.")

    # Check columns in the DataFrames before merging
    print("\nColumns in orders_df_loaded:", orders_df_loaded.columns)
    print("Columns in products_df_loaded:", products_df_loaded.columns)
    print("Columns in customers_df_loaded:", customers_df_loaded.columns)

    # Ensure 'Order ID' exists in orders_df_loaded
    if 'Order ID' not in customers_df_loaded.columns:
        customers_df_loaded['Order ID'] = orders_df_loaded['Order ID']

    # Merge Datasets
    data = pd.merge(orders_df_loaded, products_df_loaded, on='product_id', how='inner')
    data = pd.merge(data, customers_df_loaded, on='Order ID', how='inner')

    # Rename columns to avoid confusion
    data.rename(columns={'discounted_price_x': 'order_discounted_price', 
                         'discounted_price_y': 'product_discounted_price'}, inplace=True)

    # Verify columns after merging
    print("\nColumns in merged dataset:", data.columns)

    # Extract month and year from the 'date' column if it exists
    if 'date' in data.columns:
        data['order_month'] = pd.to_datetime(data['date']).dt.month
        data['order_year'] = pd.to_datetime(data['date']).dt.year
    else:
        print("The 'date' column is missing in the merged dataset.")

    # Calculate price ratio if columns are present
    if 'product_discounted_price' in data.columns and 'actual_price' in data.columns:
        data['price_ratio'] = data['product_discounted_price'] / data['actual_price']
    else:
        print("One or both of 'product_discounted_price' and 'actual_price' columns are missing in the merged dataset.")

    # Display the first few rows of the merged dataset to confirm
    print("\nMerged dataset preview:")
    print(data.head())
    
    # Saving the merged dataset for analysis
    data.to_csv('merged_data.csv', index=False)


    # Aggregate sales amount by date
    sales_trend = orders_df_loaded.groupby('date')['discounted_price'].sum().reset_index()

    # Plot Sales Trend Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(sales_trend['date'], sales_trend['discounted_price'], marker='o', linestyle='-', color='b')
    plt.title('Sales Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Sales Amount')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Top-selling products
    top_products = orders_df_loaded.groupby('product_id')['discounted_price'].sum().sort_values(ascending=False).head(10)

    # Merge with products_df to get product names
    top_products = top_products.reset_index()
    top_products = top_products.merge(products_df_loaded[['product_id', 'product_name']], on='product_id', how='left')

    # Plot Top-Selling Products
    plt.figure(figsize=(14, 8))  # Increase the figure size
    sns.barplot(x='product_name', y='discounted_price', data=top_products)
    plt.title('Top 10 Selling Products')
    plt.xlabel('Product Name')
    plt.ylabel('Total Sales Amount')
    plt.xticks(rotation=45, ha='right')  
    plt.tight_layout(pad=2) 
    plt.show()

    # Define features and target variable
    X = data[['order_month', 'order_year', 'price_ratio', 'rating']] if 'price_ratio' in data.columns else None
    y = data['product_discounted_price'] if 'product_discounted_price' in data.columns else None

    # Check if X and y are not None before proceeding
    if X is not None and y is not None:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict on the test data
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logging.info(f'Mean Squared Error: {mse}')
        logging.info(f'R2 Score: {r2}')

        # Model Interpretation
        # Coefficients of the model
        coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
        print("\nCoefficients of the model:")
        print(coefficients)

        # Visualize coefficients
        plt.figure(figsize=(8, 5))
        sns.barplot(x=coefficients.index, y='Coefficient', data=coefficients)
        plt.title('Coefficients of Linear Regression Model')
        plt.xlabel('Feature')
        plt.ylabel('Coefficient Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Saving the trained model to a file
        joblib.dump(model, 'model.pkl')
        logging.info('Model retraining completed and saved.')

    else:
        print("Features or target variable are missing. Check the dataset.")

if __name__ == '__main__':
    retrain_model()
