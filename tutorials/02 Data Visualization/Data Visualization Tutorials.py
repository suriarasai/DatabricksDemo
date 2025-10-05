# Databricks notebook source
# MAGIC %md
# MAGIC # Tutorial 2: Data Visualization
# MAGIC
# MAGIC ## Overview
# MAGIC This notebook covers various visualization techniques:
# MAGIC - Interactive charts using Databricks' built-in plotting
# MAGIC - Advanced visualizations with matplotlib, seaborn, and plotly
# MAGIC - Creating simple dashboards for insights
# MAGIC
# MAGIC **Key Libraries:**
# MAGIC - Databricks display(): Built-in interactive visualizations
# MAGIC - matplotlib: Python's foundational plotting library
# MAGIC - seaborn: Statistical data visualization
# MAGIC - plotly: Interactive, publication-quality graphs

# COMMAND ----------

# Import libraries
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.functions import col, count, sum, avg, min, max, when

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# COMMAND ----------

# Load datasets
customer_df = spark.read.csv("/Volumes/workspace/sample/datasets/customer_data.csv", header=True, inferSchema=True)
products_df = spark.read.csv("/Volumes/workspace/sample/datasets/products.csv", header=True, inferSchema=True)
sales_df = spark.read.csv("/Volumes/workspace/sample/datasets/sales_data.csv", header=True, inferSchema=True)
web_traffic_df = spark.read.csv("/Volumes/workspace/sample/datasets/web_traffic.csv", header=True, inferSchema=True)

print("Datasets loaded successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Interactive Charts with Databricks Built-in Plotting
# MAGIC
# MAGIC **Key Feature:** The `display()` function automatically creates interactive visualizations
# MAGIC
# MAGIC **Available Chart Types:**
# MAGIC - Bar charts
# MAGIC - Line charts
# MAGIC - Pie charts
# MAGIC - Scatter plots
# MAGIC - Map visualizations
# MAGIC
# MAGIC **How to Use:**
# MAGIC 1. Run `display(dataframe)`
# MAGIC 2. Click the chart icon below the table
# MAGIC 3. Select chart type and configure axes

# COMMAND ----------

# Customer distribution by segment
# After running display(), click the bar chart icon to visualize
segment_dist = customer_df.groupBy("segment").agg(
    F.count("customer_id").alias("customer_count")
).orderBy(F.desc("customer_count"))

display(segment_dist)

# COMMAND ----------

# Average income by segment
income_by_segment = customer_df.groupBy("segment").agg(
    F.avg("annual_income").alias("avg_income"),
    F.count("customer_id").alias("customer_count")
)

display(income_by_segment)

# COMMAND ----------

# Sales trend over time
# Convert date string to date type for proper time series
sales_by_date = sales_df.withColumn("date", F.to_date(col("date"))) \
    .groupBy("date").agg(
        F.sum("total_sales").alias("daily_revenue"),
        F.count("transaction_id").alias("transactions")
    ).orderBy("date")

display(sales_by_date)

# COMMAND ----------

# Regional sales performance
regional_sales = sales_df.groupBy("region").agg(
    F.sum("total_sales").alias("total_revenue"),
    F.avg("customer_satisfaction").alias("avg_satisfaction"),
    F.count("transaction_id").alias("transaction_count")
).orderBy(F.desc("total_revenue"))

display(regional_sales)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Advanced Visualization with Matplotlib
# MAGIC
# MAGIC **Matplotlib Basics:**
# MAGIC - `plt.figure()`: Create a new figure
# MAGIC - `plt.subplot()`: Create multiple plots in one figure
# MAGIC - `plt.plot()`: Line plots
# MAGIC - `plt.bar()`: Bar charts
# MAGIC - `plt.hist()`: Histograms
# MAGIC - `plt.scatter()`: Scatter plots

# COMMAND ----------

# Age distribution histogram
customer_pd = customer_df.select("age", "annual_income", "segment").toPandas()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histogram of age
axes[0].hist(customer_pd['age'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Customer Age Distribution')
axes[0].grid(True, alpha=0.3)

# Histogram of income
axes[1].hist(customer_pd['annual_income'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Annual Income ($)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Customer Income Distribution')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# Box plots for outlier detection
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Age box plot by segment
customer_pd.boxplot(column='age', by='segment', ax=axes[0])
axes[0].set_xlabel('Customer Segment')
axes[0].set_ylabel('Age')
axes[0].set_title('Age Distribution by Segment')

# Income box plot by segment
customer_pd.boxplot(column='annual_income', by='segment', ax=axes[1])
axes[1].set_xlabel('Customer Segment')
axes[1].set_ylabel('Annual Income ($)')
axes[1].set_title('Income Distribution by Segment')

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Statistical Visualizations with Seaborn
# MAGIC
# MAGIC **Seaborn Features:**
# MAGIC - Built on matplotlib with better defaults
# MAGIC - Specialized for statistical visualizations
# MAGIC - Beautiful color palettes
# MAGIC
# MAGIC **Common Plots:**
# MAGIC - `sns.countplot()`: Count plot
# MAGIC - `sns.barplot()`: Bar plot with error bars
# MAGIC - `sns.violinplot()`: Distribution shape visualization
# MAGIC - `sns.heatmap()`: Correlation matrices
# MAGIC - `sns.pairplot()`: Pairwise relationships

# COMMAND ----------

# Count plot of customer segments
plt.figure(figsize=(10, 6))
sns.countplot(data=customer_pd, x='segment', palette='viridis')
plt.title('Customer Distribution by Segment', fontsize=14, fontweight='bold')
plt.xlabel('Segment')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# Violin plot - Income distribution by segment
plt.figure(figsize=(12, 6))
sns.violinplot(data=customer_pd, x='segment', y='annual_income', palette='muted')
plt.title('Income Distribution by Customer Segment', fontsize=14, fontweight='bold')
plt.xlabel('Segment')
plt.ylabel('Annual Income ($)')
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# Scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.regplot(data=customer_pd, x='age', y='annual_income', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Age vs Annual Income Correlation', fontsize=14, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Annual Income ($)')
plt.show()

# COMMAND ----------

# Correlation heatmap
numeric_data = customer_pd[['age', 'annual_income']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_data, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Interactive Visualizations with Plotly
# MAGIC
# MAGIC **Plotly Advantages:**
# MAGIC - Fully interactive (hover, zoom, pan)
# MAGIC - Publication-quality graphics
# MAGIC - Web-based, shareable visualizations
# MAGIC
# MAGIC **Key Functions:**
# MAGIC - `px.scatter()`: Interactive scatter plots
# MAGIC - `px.line()`: Interactive line charts
# MAGIC - `px.bar()`: Interactive bar charts
# MAGIC - `px.pie()`: Interactive pie charts
# MAGIC - `go.Figure()`: Custom complex visualizations

# COMMAND ----------

# Interactive scatter plot: Age vs Income colored by segment
fig = px.scatter(customer_pd, 
                 x='age', 
                 y='annual_income', 
                 color='segment',
                 title='Customer Age vs Income by Segment',
                 labels={'age': 'Age', 'annual_income': 'Annual Income ($)'},
                 hover_data=['segment'],
                 opacity=0.6)

fig.update_layout(height=600, template='plotly_white')
fig.show()

# COMMAND ----------

# Product analysis - Price distribution by category
products_pd = products_df.select("category", "price", "rating", "brand").toPandas()

fig = px.box(products_pd, 
             x='category', 
             y='price',
             title='Product Price Distribution by Category',
             color='category')

fig.update_layout(height=600, showlegend=False, template='plotly_white')
fig.update_xaxes(tickangle=45)
fig.show()

# COMMAND ----------

# Sales trend with multiple metrics
sales_pd = sales_by_date.toPandas()

fig = go.Figure()

# Add revenue line
fig.add_trace(go.Scatter(
    x=sales_pd['date'],
    y=sales_pd['daily_revenue'],
    mode='lines+markers',
    name='Daily Revenue',
    line=dict(color='blue', width=2)
))

# Add transaction count on secondary axis
fig.add_trace(go.Scatter(
    x=sales_pd['date'],
    y=sales_pd['transactions'],
    mode='lines+markers',
    name='Transaction Count',
    yaxis='y2',
    line=dict(color='red', width=2)
))

# Update layout with secondary y-axis
fig.update_layout(
    title='Sales Performance Over Time',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Revenue ($)', side='left'),
    yaxis2=dict(title='Transactions', side='right', overlaying='y'),
    height=600,
    template='plotly_white',
    hovermode='x unified'
)

fig.show()

# COMMAND ----------

# Web traffic funnel visualization
traffic_pd = web_traffic_df.select("date", "page_views", "unique_visitors", 
                                     "sessions", "conversion_rate", "revenue").toPandas()

# Calculate conversion funnel
avg_metrics = {
    'Page Views': traffic_pd['page_views'].sum(),
    'Unique Visitors': traffic_pd['unique_visitors'].sum(),
    'Sessions': traffic_pd['sessions'].sum(),
    'Conversions': (traffic_pd['conversion_rate'].mean() * traffic_pd['sessions'].sum()) / 100
}

fig = go.Figure(go.Funnel(
    y=list(avg_metrics.keys()),
    x=list(avg_metrics.values()),
    textinfo="value+percent initial",
    marker=dict(color=["lightblue", "lightgreen", "orange", "lightcoral"])
))

fig.update_layout(
    title='Web Traffic Conversion Funnel',
    height=500,
    template='plotly_white'
)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Dashboard-Style Visualizations
# MAGIC
# MAGIC **Creating Multi-Panel Dashboards:**
# MAGIC - Combine multiple visualizations
# MAGIC - Use subplots for layout
# MAGIC - Consistent color schemes
# MAGIC - Clear titles and labels

# COMMAND ----------

# Sales Dashboard using Matplotlib
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Convert sales data to pandas for visualization
sales_viz = sales_df.toPandas()

# 1. Total Sales by Region (Top Left)
ax1 = fig.add_subplot(gs[0, :2])
region_sales = sales_viz.groupby('region')['total_sales'].sum().sort_values(ascending=False)
region_sales.plot(kind='bar', ax=ax1, color='steelblue')
ax1.set_title('Total Sales by Region', fontsize=12, fontweight='bold')
ax1.set_xlabel('Region')
ax1.set_ylabel('Revenue ($)')
ax1.tick_params(axis='x', rotation=45)

# 2. Average Satisfaction Score (Top Right)
ax2 = fig.add_subplot(gs[0, 2])
avg_satisfaction = sales_viz['customer_satisfaction'].mean()
ax2.text(0.5, 0.5, f'{avg_satisfaction:.2f}', 
         ha='center', va='center', fontsize=40, fontweight='bold', color='green')
ax2.text(0.5, 0.2, 'Avg Satisfaction', ha='center', va='center', fontsize=12)
ax2.axis('off')

# 3. Sales Distribution (Middle Left)
ax3 = fig.add_subplot(gs[1, :2])
ax3.hist(sales_viz['total_sales'], bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
ax3.set_title('Sales Amount Distribution', fontsize=12, fontweight='bold')
ax3.set_xlabel('Sale Amount ($)')
ax3.set_ylabel('Frequency')

# 4. Top Products (Middle Right)
ax4 = fig.add_subplot(gs[1, 2])
top_products = sales_viz['product'].value_counts().head(5)
top_products.plot(kind='barh', ax=ax4, color='lightgreen')
ax4.set_title('Top 5 Products', fontsize=12, fontweight='bold')
ax4.set_xlabel('Sales Count')

# 5. Sales Rep Performance (Bottom)
ax5 = fig.add_subplot(gs[2, :])
rep_performance = sales_viz.groupby('sales_rep').agg({
    'total_sales': 'sum',
    'customer_satisfaction': 'mean'
}).sort_values('total_sales', ascending=False).head(10)
rep_performance['total_sales'].plot(kind='bar', ax=ax5, color='purple', alpha=0.7)
ax5.set_title('Top 10 Sales Representatives by Revenue', fontsize=12, fontweight='bold')
ax5.set_xlabel('Sales Rep')
ax5.set_ylabel('Total Revenue ($)')
ax5.tick_params(axis='x', rotation=45)

plt.suptitle('Sales Performance Dashboard', fontsize=16, fontweight='bold', y=0.995)
plt.show()

# COMMAND ----------

# Web Traffic Dashboard using Plotly
from plotly.subplots import make_subplots

traffic_pd = web_traffic_df.toPandas()
traffic_pd['date'] = pd.to_datetime(traffic_pd['date'])

# Create subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Page Views Over Time', 'Conversion Rate Trend',
                    'Mobile vs Desktop Traffic', 'Revenue Performance'),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"type": "pie"}, {"secondary_y": False}]]
)

# 1. Page Views Over Time
fig.add_trace(
    go.Scatter(x=traffic_pd['date'], y=traffic_pd['page_views'], 
               mode='lines', name='Page Views', line=dict(color='blue')),
    row=1, col=1
)

# 2. Conversion Rate Trend
fig.add_trace(
    go.Scatter(x=traffic_pd['date'], y=traffic_pd['conversion_rate'], 
               mode='lines', name='Conversion Rate', line=dict(color='green')),
    row=1, col=2
)

# 3. Mobile Traffic Percentage (Pie Chart)
avg_mobile = traffic_pd['mobile_traffic_pct'].mean()
fig.add_trace(
    go.Pie(labels=['Mobile', 'Desktop'], 
           values=[avg_mobile, 100-avg_mobile],
           marker=dict(colors=['lightblue', 'lightgray'])),
    row=2, col=1
)

# 4. Revenue Performance
fig.add_trace(
    go.Bar(x=traffic_pd['date'], y=traffic_pd['revenue'], 
           name='Revenue', marker_color='purple'),
    row=2, col=2
)

# Update layout
fig.update_layout(
    height=800,
    title_text="Web Traffic Dashboard",
    showlegend=False,
    template='plotly_white'
)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Best Practices for Data Visualization
# MAGIC
# MAGIC **Do's:**
# MAGIC - Choose the right chart type for your data
# MAGIC - Use clear, descriptive titles and labels
# MAGIC - Keep color schemes consistent and accessible
# MAGIC - Add context with annotations when needed
# MAGIC - Make visualizations interactive when possible
# MAGIC
# MAGIC **Don'ts:**
# MAGIC - Avoid 3D charts unless necessary
# MAGIC - Don't use too many colors
# MAGIC - Avoid pie charts for more than 5-6 categories
# MAGIC - Don't truncate y-axis to mislead
# MAGIC - Avoid chart junk (unnecessary decorations)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chart Type Selection Guide
# MAGIC
# MAGIC | Data Type | Best Chart | Use Case |
# MAGIC |-----------|-----------|----------|
# MAGIC | Time Series | Line Chart | Trends over time |
# MAGIC | Categorical Comparison | Bar Chart | Compare categories |
# MAGIC | Part-to-Whole | Pie/Donut Chart | Show proportions |
# MAGIC | Distribution | Histogram | Show data spread |
# MAGIC | Relationship | Scatter Plot | Correlation analysis |
# MAGIC | Statistical | Box Plot | Show quartiles & outliers |
# MAGIC | Geographic | Map | Location-based data |
# MAGIC | Hierarchical | Treemap | Nested categories |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Takeaways
# MAGIC
# MAGIC **Databricks Built-in:**
# MAGIC - Quick and easy with `display()` function
# MAGIC - Interactive by default
# MAGIC - Good for exploratory analysis
# MAGIC
# MAGIC **Matplotlib:**
# MAGIC - Highly customizable
# MAGIC - Good for static, publication-ready charts
# MAGIC - Foundation for other libraries
# MAGIC
# MAGIC **Seaborn:**
# MAGIC - Beautiful statistical visualizations
# MAGIC - Less code than matplotlib
# MAGIC - Great for data exploration
# MAGIC
# MAGIC **Plotly:**
# MAGIC - Fully interactive
# MAGIC - Modern, web-based visualizations
# MAGIC - Great for dashboards and presentations
# MAGIC
# MAGIC **Next Steps:** Move to Notebook 3 for SQL Analytics!
