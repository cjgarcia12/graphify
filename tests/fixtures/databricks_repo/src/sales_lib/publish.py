from pyspark.sql import SparkSession


def publish():
    spark = SparkSession.builder.getOrCreate()
    revenue = spark.table("prod.gold.daily_revenue")
    revenue.write.mode("overwrite").saveAsTable("prod.reporting.revenue_snapshot")
