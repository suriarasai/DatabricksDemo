# Databricks notebook source
# MAGIC %md
# MAGIC # IoT Sensor Data Streaming with Structured Streaming
# MAGIC 
# MAGIC This notebook demonstrates real-time data processing using Spark Structured Streaming:
# MAGIC - Simulating IoT sensor data streams
# MAGIC - Real-time aggregations and anomaly detection
# MAGIC - Stream-to-stream joins
# MAGIC - Output to various sinks
# MAGIC 
# MAGIC **Use Case**: Monitor temperature sensors in real-time, detect anomalies, and trigger alerts
# MAGIC **Skills**: Structured Streaming, window operations, stream processing patterns

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Dependencies

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.streaming import StreamingQuery
import time
import threading
import random
import json
from datetime import datetime, timedelta

# Configure Spark for streaming
spark.conf.set("spark.sql.streaming.checkpointLocation", "/tmp/streaming_checkpoints")
spark.conf.set("spark.sql.adaptive.enabled", "false")  # Disable for streaming
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "false")

print("Streaming environment configured")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Generator for IoT Sensors
# MAGIC 
# MAGIC Since we're using Databricks Community Edition, we'll simulate streaming data by writing to files.

# COMMAND ----------

import threading
import json
import os
from datetime import datetime, timedelta

class IoTDataGenerator:
    """Generate simulated IoT sensor data"""
    
    def __init__(self, output_path="/tmp/iot_streaming", num_sensors=10):
        self.output_path = output_path
        self.num_sensors = num_sensors
        self.running = False
        self.thread = None
        
        # Sensor locations and normal temperature ranges
        self.sensors = {
            f"sensor_{i:03d}": {
                "location": f"Building_{chr(65 + i//3)}_Floor_{(i%3)+1}",
                "normal_temp_range": (18 + random.uniform(-2, 2), 24 + random.uniform(-2, 2)),
                "failure_prob": 0.001  # 0.1% chance of sensor malfunction per reading
            }
            for i in range(num_sensors)
        }
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
    def generate_sensor_reading(self, sensor_id, timestamp):
        """Generate a single sensor reading"""
        sensor_info = self.sensors[sensor_id]
        normal_min, normal_max = sensor_info["normal_temp_range"]
        
        # Simulate sensor malfunction
        if random.random() < sensor_info["failure_prob"]:
            # Abnormal reading - either too hot or too cold
            if random.random() < 0.5:
                temperature = random.uniform(35, 45)  # Too hot
                status = "OVERHEATING"
            else:
                temperature = random.uniform(-5, 5)   # Too cold
                status = "MALFUNCTION"
        else:
            # Normal reading with some noise
            temperature = random.uniform(normal_min, normal_max)
            # Add some daily pattern (cooler at night)
            hour = timestamp.hour
            if 6 <= hour <= 18:  # Daytime
                temperature += random.uniform(0, 2)
            else:  # Nighttime
                temperature -= random.uniform(0, 1)
            
            status = "NORMAL"
        
        # Add some random noise
        temperature += random.normalvariate(0, 0.3)
        
        return {
            "sensor_id": sensor_id,
            "location": sensor_info["location"],
            "temperature": round(temperature, 2),
            "humidity": round(random.uniform(30, 70), 1),
            "pressure": round(random.uniform(1010, 1025), 1),
            "status": status,
            "timestamp": timestamp.isoformat(),
            "reading_id": f"{sensor_id}_{int(timestamp.timestamp())}"
        }
    
    def run_generator(self, duration_minutes=5, readings_per_minute=60):
        """Run the data generator for specified duration"""
        self.running = True
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        file_counter = 0
        
        print(f"Starting IoT data generation for {duration_minutes} minutes...")
        print(f"Writing to: {self.output_path}")
        
        while self.running and datetime.now() < end_time:
            # Generate batch of readings
            current_time = datetime.now()
            batch_data = []
            
            for sensor_id in self.sensors.keys():
                reading = self.generate_sensor_reading(sensor_id, current_time)
                batch_data.append(reading)
            
            # Write batch to file
            filename = f"iot_batch_{file_counter:06d}.json"
            filepath = os.path.join(self.output_path, filename)
            
            with open(filepath, 'w') as f:
                for reading in batch_data:
                    f.write(json.dumps(reading) + '\n')
            
            file_counter += 1
            
            # Wait before next batch
            time.sleep(60 / readings_per_minute)
        
        print("Data generation completed")
    
    def start_async(self, duration_minutes=10):
        """Start data generation in background thread"""
        if self.thread and self.thread.is_alive():
            print("Generator already running")
            return
            
        self.thread = threading.Thread(
            target=self.run_generator, 
            args=(duration_minutes,)
        )
        self.thread.daemon = True
        self.thread.start()
        print("IoT data generator started in background")
    
    def stop(self):
        """Stop the data generation"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("Data generator stopped")

# Initialize and start the generator
generator = IoTDataGenerator()
generator.start_async(duration_minutes=15)  # Run for 15 minutes

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define Schema for Streaming Data

# COMMAND ----------

# Define the schema for IoT sensor data
iot_schema = StructType([
    StructField("sensor_id", StringType(), True),
    StructField("location", StringType(), True),
    StructField("temperature", DoubleType(), True),
    StructField("humidity", DoubleType(), True),
    StructField("pressure", DoubleType(), True),
    StructField("status", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("reading_id", StringType(), True)
])

print("Schema defined for IoT streaming data")
iot_schema

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Streaming DataFrame

# COMMAND ----------

# Create streaming DataFrame
streaming_df = spark \
    .readStream \
    .schema(iot_schema) \
    .option("maxFilesPerTrigger", 1) \
    .json("/tmp/iot_streaming/")

# Convert timestamp string to timestamp type and add processing time
streaming_df = streaming_df \
    .withColumn("event_timestamp", to_timestamp(col("timestamp"))) \
    .withColumn("processing_time", current_timestamp()) \
    .drop("timestamp")

print("Streaming DataFrame created")
print("Streaming:", streaming_df.isStreaming)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Real-time Aggregations

# COMMAND ----------

# Window-based aggregations - count readings per sensor per minute
windowed_counts = streaming_df \
    .withWatermark("event_timestamp", "2 minutes") \
    .groupBy(
        window(col("event_timestamp"), "1 minute"),
        col("sensor_id"),
        col("location")
    ) \
    .agg(
        count("*").alias("reading_count"),
        avg("temperature").alias("avg_temperature"),
        max("temperature").alias("max_temperature"),
        min("temperature").alias("min_temperature"),
        stddev("temperature").alias("temp_stddev")
    )

print("Windowed aggregations defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Anomaly Detection

# COMMAND ----------

# Define anomaly detection logic
def detect_anomalies(batch_df, batch_id):
    """Custom function to detect temperature anomalies"""
    
    if batch_df.count() == 0:
        return
    
    print(f"Processing batch {batch_id} with {batch_df.count()} records")
    
    # Find temperature anomalies
    anomalies = batch_df.filter(
        (col("temperature") > 30) |   # Too hot
        (col("temperature") < 10) |   # Too cold
        (col("status") != "NORMAL")   # System reported anomaly
    )
    
    if anomalies.count() > 0:
        print(f"ALERT: Found {anomalies.count()} anomalies in batch {batch_id}")
        anomalies.select("sensor_id", "location", "temperature", "status", "event_timestamp").show()
        
        # In production, this could trigger alerts, send notifications, etc.
        # For demo, we'll just save to a table
        anomalies.write \
            .mode("append") \
            .option("path", "/tmp/anomaly_alerts") \
            .saveAsTable("anomaly_alerts")

# Anomaly detection stream
anomaly_stream = streaming_df \
    .writeStream \
    .foreachBatch(detect_anomalies) \
    .outputMode("append") \
    .option("checkpointLocation", "/tmp/anomaly_checkpoints") \
    .trigger(processingTime="30 seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Start Streaming Queries

# COMMAND ----------

# Start the anomaly detection stream
print("Starting anomaly detection stream...")
anomaly_query = anomaly_stream.start()

# Wait a bit for data to start flowing
time.sleep(10)

# COMMAND ----------

# Start windowed aggregation stream with console output
print("Starting windowed aggregation stream...")

windowed_query = windowed_counts \
    .writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .option("checkpointLocation", "/tmp/windowed_checkpoints") \
    .trigger(processingTime="1 minute") \
    .start()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Real-time Dashboard Data

# COMMAND ----------

# Create a stream that maintains current sensor status
current_status = streaming_df \
    .select("sensor_id", "location", "temperature", "humidity", "pressure", "status", "event_timestamp") \
    .writeStream \
    .outputMode("append") \
    .format("memory") \
    .queryName("current_sensor_readings") \
    .trigger(processingTime="10 seconds") \
    .start()

# Let it run for a bit
time.sleep(30)

# COMMAND ----------

# Query the in-memory table for dashboard
print("Current Sensor Readings (Last 30 seconds):")
current_readings = spark.sql("""
    SELECT 
        sensor_id,
        location,
        temperature,
        humidity,
        status,
        event_timestamp
    FROM current_sensor_readings
    WHERE event_timestamp > current_timestamp() - INTERVAL 30 SECONDS
    ORDER BY event_timestamp DESC
""")

display(current_readings)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Advanced Stream Processing

# COMMAND ----------

# Calculate moving averages and detect trends
moving_averages = streaming_df \
    .withWatermark("event_timestamp", "5 minutes") \
    .groupBy(
        col("sensor_id"),
        window(col("event_timestamp"), "5 minutes", "1 minute")
    ) \
    .agg(
        avg("temperature").alias("avg_temp"),
        count("*").alias("reading_count")
    ) \
    .withColumn("window_start", col("window.start")) \
    .withColumn("window_end", col("window.end")) \
    .drop("window")

# Add trend detection
trend_detection = moving_averages \
    .withColumn(
        "prev_avg_temp",
        lag("avg_temp", 1).over(
            Window.partitionBy("sensor_id").orderBy("window_start")
        )
    ) \
    .withColumn(
        "temp_trend",
        when(col("avg_temp") > col("prev_avg_temp") + 2, "RISING")
        .when(col("avg_temp") < col("prev_avg_temp") - 2, "FALLING")
        .otherwise("STABLE")
    )

print("Advanced analytics stream defined")

# COMMAND ----------

# Start the trend detection stream
trend_query = trend_detection \
    .writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .option("checkpointLocation", "/tmp/trend_checkpoints") \
    .trigger(processingTime="2 minutes") \
    .start()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Stream Monitoring and Management

# COMMAND ----------

# Function to display stream status
def display_stream_status():
    """Display status of all active streams"""
    active_streams = spark.streams.active
    
    print(f"Active Streams: {len(active_streams)}")
    print("=" * 50)
    
    for stream in active_streams:
        print(f"Stream ID: {stream.id}")
        print(f"Name: {stream.name}")
        print(f"Status: {stream.status}")
        
        if hasattr(stream, 'lastProgress'):
            progress = stream.lastProgress
            if progress:
                print(f"Batch ID: {progress.get('batchId', 'N/A')}")
                print(f"Input Rows: {progress.get('inputRowsPerSecond', 'N/A')}")
                print(f"Processing Rate: {progress.get('processingRate', 'N/A')}")
        
        print("-" * 30)

# Monitor streams
display_stream_status()

# COMMAND ----------

# Let the streams run for a few minutes to collect data
print("Letting streams run for 3 minutes to collect data...")
time.sleep(180)

# Check final status
display_stream_status()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Stream Results Analysis

# COMMAND ----------

# Check if we have any anomaly alerts
try:
    anomaly_count = spark.sql("SELECT COUNT(*) as count FROM anomaly_alerts").collect()[0]['count']
    print(f"Total anomalies detected: {anomaly_count}")
    
    if anomaly_count > 0:
        print("Sample anomalies:")
        display(spark.sql("""
            SELECT sensor_id, location, temperature, status, event_timestamp
            FROM anomaly_alerts 
            ORDER BY event_timestamp DESC 
            LIMIT 10
        """))
except Exception as e:
    print("No anomalies table found - this is normal if no anomalies were detected")

# COMMAND ----------

# Check current sensor readings summary
print("Summary of recent sensor readings:")
recent_summary = spark.sql("""
    SELECT 
        sensor_id,
        location,
        COUNT(*) as total_readings,
        AVG(temperature) as avg_temperature,
        MAX(temperature) as max_temperature,
        MIN(temperature) as min_temperature,
        MAX(event_timestamp) as last_reading
    FROM current_sensor_readings
    GROUP BY sensor_id, location
    ORDER BY sensor_id
""")

display(recent_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Cleanup and Stop Streams

# COMMAND ----------

# Function to safely stop all streams
def stop_all_streams():
    """Stop all active streaming queries"""
    active_streams = spark.streams.active
    
    print(f"Stopping {len(active_streams)} active streams...")
    
    for stream in active_streams:
        try:
            print(f"Stopping stream: {stream.name or stream.id}")
            stream.stop()
            print(f"✓ Stream stopped successfully")
        except Exception as e:
            print(f"✗ Error stopping stream: {e}")
    
    print("All streams stopped")

# Stop the data generator
generator.stop()

# Stop all streaming queries
stop_all_streams()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Insights and Learning Outcomes
# MAGIC 
# MAGIC ### Streaming Concepts Demonstrated:
# MAGIC 1. **Structured Streaming**: Modern stream processing with DataFrame API
# MAGIC 2. **Windowed Operations**: Time-based aggregations and analytics
# MAGIC 3. **Watermarks**: Handling late-arriving data
# MAGIC 4. **Stateful Processing**: Maintaining state across micro-batches
# MAGIC 5. **Multiple Output Modes**: Append, complete, and update modes
# MAGIC 6. **Stream Monitoring**: Tracking performance and progress
# MAGIC 
# MAGIC ### Real-world Applications:
# MAGIC - **IoT Monitoring**: Real-time sensor data processing
# MAGIC - **Financial Trading**: Market data analysis and alerting
# MAGIC - **Web Analytics**: Live user behavior tracking
# MAGIC - **Manufacturing**: Equipment monitoring and predictive maintenance
# MAGIC - **Social Media**: Real-time sentiment analysis
# MAGIC 
# MAGIC ### Technical Skills Covered:
# MAGIC - Stream ingestion from files (simulating Kafka/Kinesis)
# MAGIC - Complex event processing and anomaly detection
# MAGIC - Window functions and time-based aggregations
# MAGIC - Stream-to-batch integration
# MAGIC - Error handling and fault tolerance
# MAGIC - Performance monitoring and optimization
# MAGIC 
# MAGIC ### Production Considerations:
# MAGIC 1. **Checkpointing**: Fault tolerance and exactly-once processing
# MAGIC 2. **Backpressure**: Handle varying data rates
# MAGIC 3. **Schema Evolution**: Handle changing data structures
# MAGIC 4. **Monitoring**: Track lag, throughput, and errors
# MAGIC 5. **Scaling**: Partition strategies and cluster sizing
# MAGIC 
# MAGIC This streaming example provides a foundation for building real-time analytics solutions using Databricks and Spark Structured Streaming.

# COMMAND ----------