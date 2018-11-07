package nlp_spark
//http://lxw1234.com/archives/2016/01/605.htm

import org.apache.spark.ml.feature.{HashingTF, IDF, LabeledPoint, Tokenizer}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{Row, SparkSession}

object naive_bayes_data_preparation {
    case class RawDataRecord(category: String, text: String)

    val spark = SparkSession
      .builder()
      .master("yarn-client")
      .appName("Naive-Bayes")
      .getOrCreate()

    import spark.implicits._

    var srcDF = spark.sparkContext.textFile("1.txt")
      .map{x =>
          var data = x.split(",")
          RawDataRecord(data(0), data(1))
      }.toDF()

    srcDF.select("category", "text").take(2).foreach(println)

    //convert the segmented words into array
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    var wordsData = tokenizer.transform(srcDF)

    wordsData.select($"category", $"text", $"words").take(2).foreach(println)

    //convert every word into Int, and calculate TF
    var hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(100)
    var featurizedData = hashingTF.transform(wordsData)

    featurizedData.select($"category", $"words", $"rawFeatures").take(2).foreach(println)
    /**
      * [0,WrappedArray(苹果, 官网, 苹果, 宣布),(100,[23,81,96],[2.0,1.0,1.0])]
      * [1,WrappedArray(苹果, 梨, 香蕉),(100,[23,72,92],[1.0,1.0,1.0])]
      */

    /**
      * TF-IDF
      */
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.select($"vategory", $"words", $"features").take(2).foreach(println)
    /**
      * [0,WrappedArray(苹果, 官网, 苹果, 宣布),(100,[23,81,96],[0.0,0.4054651081081644,0.4054651081081644])]
      * [1,WrappedArray(苹果, 梨, 香蕉),(100,[23,72,92],[0.0,0.4054651081081644,0.4054651081081644])]
      */

    val trainDataRDD = rescaledData.select($"category", $"features").map{
        case Row(label: String, features: Vector) =>
            LabeledPoint(label.toDouble, Vectors.dense(features.toArray))
    }
}
