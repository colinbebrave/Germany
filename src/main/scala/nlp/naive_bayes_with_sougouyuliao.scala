package nlp

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DoubleType


object naive_bayes_with_sougouyuliao {

    def main(args: Array[String]): Unit = {
        val spark = SparkSession
          .builder
          .master("local[*]")
          .appName("training Naive Bayes model")
          .getOrCreate()

        import spark.implicits._

        val srcDF = spark.sparkContext.textFile("/Users/wangqi/Documents/Java/Germany/src/main/scala/nlp/sougou_yuliao/").map{x =>
            var data = x.split(",")
            (data(0), data(1))
        }.toDF("category", "text")

        srcDF.select('category).distinct().show()

        val resData = srcDF.withColumn("category_tmp", srcDF("category").cast(DoubleType)).drop("category").withColumnRenamed("category_tmp", "label")

        //将词语转换成数组
        val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
        var wordsData = tokenizer.transform(resData)
        println("output1: ")
        wordsData.select($"category", $"text", $"words").take(1)

        //计算每个词在文档中的词频
        var hashingTF = new HashingTF().setNumFeatures(500000).setInputCol("words").setOutputCol("rawFeatures")
        val featurizedData = hashingTF.transform(wordsData)
        println("output2: ")
        featurizedData.select($"category", $"words", $"rawFeatures").take(1)

        //计算每个词的TF-IDF
        val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
        val idfModel = idf.fit(featurizedData)
        val rescaledData = idfModel.transform(featurizedData)

        val splits = rescaledData.randomSplit(Array(0.7, 0.3))
        val Array(trainingDF, testingDF) = Array(splits(0), splits(1))

        //训练NaiveBayes
        val nb = new NaiveBayes().setLabelCol("label").setFeaturesCol("features").fit(trainingDF)

        val predictions = nb.transform(testingDF)
        predictions.show()

        val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
        val accuracy = evaluator.evaluate(predictions)

        println(s"Test set accuracy = $accuracy")
        spark.stop()
    }
}
