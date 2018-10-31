package spark_streaming.apache_spark_2_x

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.StreamingLogisticRegressionWithSGD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.streaming.{Seconds, StreamingContext}

import scala.collection.mutable.Queue


object StreamingLogistic {
    def main(args: Array[String]): Unit = {
        Logger.getLogger("org").setLevel(Level.ERROR)
        Logger.getRootLogger.setLevel(Level.WARN)

        val spark = SparkSession
          .builder
          .master("local[*]")
          .appName("Logistic Streaming App")
          .config("spark.sql.warehouse.dir", ".")
          .getOrCreate()

        import spark.implicits._

        val ssc = new StreamingContext(spark.sparkContext, Seconds(2))

        val rawDF = spark.read.text("pima-indians-diabetes.data").as[String]

        val buf = rawDF.rdd.map(value => {
            val data = value.split(",")
            (data.init.toSeq, data.last)
        })

        val lps = buf.map {case (feature: Seq[String], label: String) =>
        val featureVector = feature.map(_.toDouble).toArray[Double]
        LabeledPoint(label.toDouble, Vectors.dense(featureVector))}

        val trainQueue = new Queue[RDD[LabeledPoint]]()
        val testQueue = new Queue[RDD[LabeledPoint]]()

        val trainingStream = ssc.queueStream(trainQueue)
        val testStream = ssc.queueStream(testQueue)

        val numFeatures = 8

        val model = new StreamingLogisticRegressionWithSGD()
          .setInitialWeights(Vectors.zeros(numFeatures))
          .setNumIterations(15)
          .setStepSize(0.5)
          .setMiniBatchFraction(0.25)

        model.trainOn(trainingStream)
        val result = model.predictOnValues(testStream.map(lp => (lp.label, lp.features)))

        result.map{ case (label: Double, prediction: Double) => (label, prediction)}.print()

        ssc.start()

        val Array(train, test) = lps.randomSplit(Array(0.8, 0.2))

        trainQueue += train
        Thread.sleep(4000)

        val testGroups = test.randomSplit(Array(0.5, 0.5))
        testGroups.foreach(group => {
            testQueue += group
            Thread.sleep(2000)
        })
        ssc.stop()
    }
}
