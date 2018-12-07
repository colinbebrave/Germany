package example

import org.apache.spark.sql.SparkSession

object ctr_prediction_using_GBDT {
    def main(args: Array[String]): Unit = {
        val spark = SparkSession
          .builder
          .master("local[*]")
          .appName("CTR prediction using GBDT")
          .getOrCreate()

        val ctrDF = spark.read
          .option("delimiter", ",")
          .option("header", "true")
          .option("inferschema", "true")
          .csv("/Users/wangqi/Downloads/test.txt")
    }
}
