"""
Data Utilities for Databricks Demo
==================================

Common utility functions for data processing, validation, and transformations
used across multiple notebooks and examples.

Author: Databricks Demo Team
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import random
from datetime import datetime, timedelta


class DataGenerator:
    """Generate synthetic data for demo purposes"""
    
    @staticmethod
    def generate_customer_data(spark, num_customers=1000, seed=42):
        """Generate synthetic customer data"""
        random.seed(seed)
        
        customers = []
        first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
        
        for i in range(num_customers):
            customer_id = f"CUST_{i+1:06d}"
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            email = f"{first_name.lower()}.{last_name.lower()}@email.com"
            age = random.randint(21, 70)
            signup_date = datetime.now() - timedelta(days=random.randint(30, 1095))
            
            customers.append((customer_id, first_name, last_name, email, age, signup_date))
        
        schema = StructType([
            StructField("customer_id", StringType(), True),
            StructField("first_name", StringType(), True),
            StructField("last_name", StringType(), True),
            StructField("email", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("signup_date", TimestampType(), True)
        ])
        
        return spark.createDataFrame(customers, schema)
    
    @staticmethod
    def generate_sales_data(spark, num_transactions=5000, seed=42):
        """Generate synthetic sales transaction data"""
        random.seed(seed)
        
        transactions = []
        products = ["Laptop", "Smartphone", "Tablet", "Headphones", "Camera", "Watch"]
        categories = ["Electronics", "Computers", "Mobile", "Audio", "Photography", "Wearable"]
        
        for i in range(num_transactions):
            transaction_id = f"TXN_{i+1:08d}"
            customer_id = f"CUST_{random.randint(1, 1000):06d}"
            product_idx = random.randint(0, len(products)-1)
            product = products[product_idx]
            category = categories[product_idx]
            
            quantity = random.randint(1, 5)
            unit_price = round(random.uniform(50, 1500), 2)
            total_amount = quantity * unit_price
            
            transaction_date = datetime.now() - timedelta(days=random.randint(1, 365))
            
            transactions.append((
                transaction_id, customer_id, product, category,
                quantity, unit_price, total_amount, transaction_date
            ))
        
        schema = StructType([
            StructField("transaction_id", StringType(), True),
            StructField("customer_id", StringType(), True),
            StructField("product", StringType(), True),
            StructField("category", StringType(), True),
            StructField("quantity", IntegerType(), True),
            StructField("unit_price", DoubleType(), True),
            StructField("total_amount", DoubleType(), True),
            StructField("transaction_date", TimestampType(), True)
        ])
        
        return spark.createDataFrame(transactions, schema)


class DataQualityChecker:
    """Data quality assessment utilities"""
    
    @staticmethod
    def assess_data_quality(df, table_name="DataFrame"):
        """Comprehensive data quality assessment"""
        print(f"\n=== Data Quality Report for {table_name} ===")
        
        total_rows = df.count()
        print(f"Total rows: {total_rows:,}")
        
        # Schema information
        print(f"\nSchema:")
        df.printSchema()
        
        # Null analysis
        print(f"\nNull Analysis:")
        for column in df.columns:
            null_count = df.filter(col(column).isNull()).count()
            null_percentage = (null_count / total_rows) * 100
            print(f"  {column}: {null_count:,} nulls ({null_percentage:.2f}%)")
        
        # Duplicate analysis
        distinct_rows = df.distinct().count()
        duplicates = total_rows - distinct_rows
        print(f"\nDuplicate Analysis:")
        print(f"  Distinct rows: {distinct_rows:,}")
        print(f"  Duplicate rows: {duplicates:,}")
        
        # Basic statistics for numeric columns
        numeric_columns = [f.name for f in df.schema.fields 
                          if f.dataType in [IntegerType(), LongType(), FloatType(), DoubleType()]]
        
        if numeric_columns:
            print(f"\nNumeric Columns Statistics:")
            display(df.select(numeric_columns).describe())
        
        return {
            "total_rows": total_rows,
            "distinct_rows": distinct_rows,
            "duplicates": duplicates,
            "null_counts": {col: df.filter(col(col).isNull()).count() for col in df.columns}
        }
    
    @staticmethod
    def validate_data_ranges(df, column_ranges):
        """Validate that data falls within expected ranges"""
        issues = []
        
        for column, (min_val, max_val) in column_ranges.items():
            if column in df.columns:
                out_of_range = df.filter(
                    (col(column) < min_val) | (col(column) > max_val)
                ).count()
                
                if out_of_range > 0:
                    issues.append(f"{column}: {out_of_range} values outside range [{min_val}, {max_val}]")
        
        if issues:
            print("Data Range Validation Issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("All data ranges validated successfully!")
        
        return len(issues) == 0


class DataTransformer:
    """Common data transformation utilities"""
    
    @staticmethod
    def add_time_features(df, date_column):
        """Add common time-based features from a date column"""
        return df.withColumn("year", year(col(date_column))) \
                .withColumn("month", month(col(date_column))) \
                .withColumn("day_of_week", dayofweek(col(date_column))) \
                .withColumn("quarter", quarter(col(date_column))) \
                .withColumn("is_weekend", when(dayofweek(col(date_column)).isin([1, 7]), 1).otherwise(0))
    
    @staticmethod
    def calculate_rfm_metrics(df, customer_col, date_col, amount_col):
        """Calculate RFM (Recency, Frequency, Monetary) metrics"""
        from pyspark.sql.functions import datediff, current_date
        
        rfm = df.groupBy(customer_col).agg(
            datediff(current_date(), max(col(date_col))).alias("recency_days"),
            countDistinct(col(date_col)).alias("frequency"),
            sum(col(amount_col)).alias("monetary_value")
        )
        
        return rfm
    
    @staticmethod
    def create_feature_buckets(df, column, bucket_boundaries, bucket_labels=None):
        """Create categorical buckets from continuous variables"""
        if bucket_labels is None:
            bucket_labels = [f"bucket_{i}" for i in range(len(bucket_boundaries) + 1)]
        
        # Create bucketizer conditions
        conditions = []
        for i, boundary in enumerate(bucket_boundaries):
            if i == 0:
                conditions.append(when(col(column) <= boundary, bucket_labels[i]))
            elif i == len(bucket_boundaries) - 1:
                conditions.append(when(col(column) <= boundary, bucket_labels[i]))
            else:
                conditions.append(when(col(column) <= boundary, bucket_labels[i]))
        
        # Add final condition for values above the last boundary
        conditions.append(lit(bucket_labels[-1]))
        
        # Chain conditions
        bucket_column = conditions[0]
        for condition in conditions[1:]:
            bucket_column = bucket_column.otherwise(condition)
        
        return df.withColumn(f"{column}_bucket", bucket_column)


class SparkOptimizer:
    """Spark performance optimization utilities"""
    
    @staticmethod
    def optimize_dataframe(df, cache=True, repartition=None):
        """Apply common optimizations to a DataFrame"""
        if repartition:
            df = df.repartition(repartition)
        
        if cache:
            df = df.cache()
        
        return df
    
    @staticmethod
    def analyze_partitions(df):
        """Analyze DataFrame partitioning"""
        num_partitions = df.rdd.getNumPartitions()
        partition_sizes = df.glom().map(len).collect()
        
        print(f"Number of partitions: {num_partitions}")
        print(f"Partition sizes: {partition_sizes}")
        print(f"Total records: {sum(partition_sizes)}")
        print(f"Average partition size: {sum(partition_sizes) / len(partition_sizes):.1f}")
        print(f"Min partition size: {min(partition_sizes)}")
        print(f"Max partition size: {max(partition_sizes)}")
        
        return {
            "num_partitions": num_partitions,
            "partition_sizes": partition_sizes,
            "total_records": sum(partition_sizes)
        }


def get_spark_session(app_name="DatabricksDemo"):
    """Get or create SparkSession with optimized configuration"""
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()


# Example usage and testing functions
if __name__ == "__main__":
    # This section would run if the script is executed directly
    # (not typically the case in Databricks notebooks)
    
    spark = get_spark_session()
    
    # Generate sample data
    customers = DataGenerator.generate_customer_data(spark, 100)
    sales = DataGenerator.generate_sales_data(spark, 500)
    
    # Assess data quality
    DataQualityChecker.assess_data_quality(customers, "Customers")
    DataQualityChecker.assess_data_quality(sales, "Sales")
    
    # Apply transformations
    sales_with_time = DataTransformer.add_time_features(sales, "transaction_date")
    
    print("Utility functions loaded successfully!")