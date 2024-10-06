# Amazon Sales Data Analysis Project

## Project Overview

This project is part of my internship at **Unified Mentors**, where I am working on analyzing sales data from Amazon's platform to extract valuable insights into product performance, customer behavior, and pricing strategies. The primary goal is to use data analysis to understand sales trends, customer preferences, and product performance, and to build predictive models for forecasting future sales. This project involves cleaning and exploring the data, creating new features, and developing an interactive dashboard using Tableau to visualize key metrics and insights.

## Objectives

The primary objectives of this project are:
1. Analyze sales performance trends over time.
2. Identify top N-selling products.
3. Understand customer distribution across locations.
4. Examine product rating patterns.
5. Investigate pricing strategies and discount impacts.
6. Build a predictive model for future sales performance.

## Tools and Technologies Used

- **Programming Languages**: Python (pandas, numpy, matplotlib, seaborn, scikit-learn)
- **Data Visualization**: Tableau
- **Web Framework**: Flask

## Dataset Description

The project utilizes a comprehensive Amazon sales dataset containing the following key information:
- **Product Information**: `product_id`, `product_name`, `category`
- **Pricing Information**: `discounted_price`, `actual_price`, `discount_percentage`
- **Customer Details**: `user_id`, `user_name`, `review_id`, `review_title`, `review_content`, `location`
- **Rating and Review Data**: `rating`, `rating_count`
- **Additional Information**: `img_link`, `product_link`

## Data Preparation

The data preparation process involved the following steps:
1. Data type conversion to ensure numeric formats.
2. Handling missing values using the forward fill method.
3. Simulated dataset creation for orders and customers.
4. Dataset merging for a comprehensive view.
5. Feature engineering to extract useful insights.

## Exploratory Data Analysis (EDA)

The EDA phase involved:
- Numerical feature analysis through histograms.
- Sales trends analysis with line charts.
- Identifying top-selling products and customer distribution.

## Predictive Modeling

A predictive model was developed to forecast `product_discounted_price` based on various features, including:
- `order_month`
- `order_year`
- `price_ratio`
- `rating`

### Model Implementation
1. **Model Selection**: Linear Regression
2. **Data Splitting**: 80% training, 20% testing
3. **Model Evaluation**: Mean Squared Error (MSE) and R-squared (R²) score.

## Deployment

A web application was developed and deployed to make the insights and predictive capabilities accessible to stakeholders. The web application includes:
- A dashboard for visualizing sales trends.
- A product recommendation system.
- A sales forecast tool using the trained predictive model.

## Tableau Dashboard

An interactive Tableau dashboard was created to visualize key insights, including:
- Sales Overview
- Customer Location Distribution
- Sales Trends
- Product Analysis
- Price Analysis

## Logging and Monitoring

To ensure continued performance, a logging and monitoring system captures application events, errors, and user interactions.

## Internship Acknowledgment

This project is part of my internship with Unified Mentors, where I am gaining hands-on experience in data science, web development, and predictive modeling. During the internship, I’ve developed this project to analyze Amazon sales data and create a robust system for data-driven decision-making.

## Conclusion

The **Amazon Sales Data Analysis** project successfully progressed from initial data analysis to a fully deployed and monitored predictive system. This project demonstrates the power of data science in e-commerce, providing valuable insights for data-driven decision-making.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- Tableau (for dashboard visualization)
- Flask (for web application development)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/carinadesouza/Analyzing-Amazon-Sales-data-Project.git
2.  Navigate to the project directory:
```bash
  cd amazon-sales-analysis
```

3.Install the required packages:
```bash
pip install -r requirements.txt
```

4.Start the Flask application:
```bash
python app.py
Access the application at http://127.0.0.1:5000.
```



