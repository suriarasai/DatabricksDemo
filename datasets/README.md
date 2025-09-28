# Sample Datasets for Databricks Tutorials

This directory contains sample datasets used throughout the DBSamples tutorials. All datasets are synthetic and created for educational purposes.

## Available Datasets

### 1. Sales Data (sales_data.csv)
- **Description**: E-commerce sales transactions
- **Records**: ~1,000 rows
- **Columns**: date, product, region, sales_rep, quantity, unit_price, total_sales, customer_satisfaction
- **Use Cases**: Data exploration, visualization, time series analysis

### 2. Customer Data (customer_data.csv)
- **Description**: Customer demographics and behavior
- **Records**: ~500 rows  
- **Columns**: customer_id, name, email, registration_date, city, state, age, segment
- **Use Cases**: Customer analytics, segmentation, geographic analysis

### 3. Time Series Data (web_traffic.csv)
- **Description**: Website traffic and engagement metrics
- **Records**: ~365 rows (daily data for one year)
- **Columns**: date, page_views, unique_visitors, bounce_rate, conversion_rate, revenue
- **Use Cases**: Time series forecasting, trend analysis

### 4. Product Catalog (products.csv)
- **Description**: Product information and pricing
- **Records**: ~200 rows
- **Columns**: product_id, name, category, price, cost, supplier, launch_date
- **Use Cases**: Inventory analysis, profitability analysis, product performance

## Data Generation

All datasets are programmatically generated using Python with realistic patterns and distributions. The generation scripts are included in the tutorial notebooks.

## Usage Instructions

### Loading in Databricks

```python
# Read CSV files from DBFS
df = spark.read.option("header", "true").option("inferSchema", "true").csv("/FileStore/shared_uploads/your_email/sales_data.csv")

```

### Uploading to Databricks Free Edition

1. Go to your Databricks workspace
2. Click on "Data" in the left sidebar
3. Click "Create Table"
4. Select "Upload File"
5. Choose the CSV file from this repository
6. Follow the import wizard to create a table

## Data Dictionary

### Sales Data Fields
- `date`: Transaction date (YYYY-MM-DD)
- `product`: Product name/type
- `region`: Geographic region (North, South, East, West)
- `sales_rep`: Sales representative ID
- `quantity`: Number of items sold
- `unit_price`: Price per unit ($)
- `total_sales`: Total transaction value ($)
- `customer_satisfaction`: Rating 1-5

### Customer Data Fields
- `customer_id`: Unique customer identifier
- `name`: Customer full name
- `email`: Email address
- `registration_date`: Account creation date
- `city`: Customer city
- `state`: Customer state/province
- `age`: Customer age
- `segment`: Customer tier (Premium, Standard, Basic)

### Web Traffic Fields
- `date`: Date of measurement
- `page_views`: Total page views
- `unique_visitors`: Unique visitor count
- `bounce_rate`: Percentage of single-page sessions
- `conversion_rate`: Percentage of visitors who converted
- `revenue`: Daily revenue generated ($)

### Product Catalog Fields
- `product_id`: Unique product identifier
- `name`: Product name
- `category`: Product category
- `price`: Selling price ($)
- `cost`: Cost of goods sold ($)
- `supplier`: Supplier name
- `launch_date`: Product launch date

## License

These datasets are released under CC0 1.0 Universal license - free to use for any purpose.