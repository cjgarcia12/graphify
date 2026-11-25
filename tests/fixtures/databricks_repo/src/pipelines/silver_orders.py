import dlt
from pyspark.sql import functions as F


@dlt.table(name="orders_clean", comment="Deduplicated orders")
def orders_clean():
    catalog = spark.conf.get("catalog")
    return (
        spark.readStream.table(f"{catalog}.bronze.orders_raw")
        .dropDuplicates(["order_id"])
    )


@dlt.table
def customers():
    return spark.read.table("dev.bronze.customers_raw")


@dlt.table
def orders_enriched():
    return dlt.read("orders_clean").join(dlt.read("customers"), "customer_id")
