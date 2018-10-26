package spark_in_action

import org.apache.spark.sql.SparkSession

object chapter8_ml_classification_and_clustering {
    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("Spark SQL basic example")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()
    val census_raw = spark.sparkContext.textFile("/Users/wangqi/Desktop/people.txt", 4).
      map(x => x.split(", ")).
      map(row => row.map(x => try { x.toDouble }
      catch { case _ : Throwable => x }))

}

