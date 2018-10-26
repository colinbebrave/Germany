package Scala_and_Spark_for_Big_Data_Analysis
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.LDA

object chapter15_text_classification {
    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("feature extraction")
      .getOrCreate()
    import spark.implicits._

}
