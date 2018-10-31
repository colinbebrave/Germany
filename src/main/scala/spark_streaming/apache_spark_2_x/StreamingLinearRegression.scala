package spark_streaming.apache_spark_2_x

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, StreamingLinearRegressionWithSGD}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.streaming.{Seconds, StreamingContext}

import scala.collection.mutable.Queue

object StreamingLinearRegression {
    def main(args: Array[String]): Unit = {
        Logger.getLogger("org").setLevel(Level.ERROR)

        val spark = SparkSession
          .builder()
          .master("local[*]")
          .appName("Regression StreamingApp")
          .config("spark.sql.warehouse.dir", ".")
          .config("spark.executor.memory", "2g")
          .getOrCreate()
        val ssc = new StreamingContext(spark.sparkContext, Seconds(2))

        Logger.getRootLogger.setLevel(Level.WARN)

        val rawDF = spark.read
          .format("com.databricks.spark.csv")
          .option("inferschema", "true")
          .option("header", "true")
          .option("delimiter", ",")
          .load("winequality-white.csv")

        val rdd = rawDF.rdd.zipWithUniqueId()

        rdd.collect().foreach(println)

        val lookupQuality = rdd.map{case (r: Row, id: Long) => (id, r.getInt(11))}.collect().toMap

        lookupQuality.foreach(println)

        val d = rdd.map{case (r: Row, id: Long) => LabeledPoint(id,
            Vectors.dense(r.getDouble(0), r.getDouble(1), r.getDouble(2), r.getDouble(3), r.getDouble(4),
                r.getDouble(5),r.getDouble(6),r.getDouble(7),r.getDouble(8),r.getDouble(9),r.getDouble(10)))}

        rdd.collect().foreach(println)

        val trainQueue = new Queue[RDD[LabeledPoint]]()
        val testQueue = new Queue[RDD[LabeledPoint]]()

        val trainingStream = ssc.queueStream(trainQueue)
        val testStream = ssc.queueStream(testQueue)

        val numFeatures = 11
        val model = new StreamingLinearRegressionWithSGD()
          .setInitialWeights(Vectors.zeros(numFeatures))
          .setNumIterations(25)
          .setStepSize(0.1)
          .setMiniBatchFraction(0.25)

        model.trainOn(trainingStream)
        val result = model.predictOnValues(testStream.map(lp => (lp.label, lp.features)))
        result.map{case(id: Double, prediction: Double) => (id, prediction, lookupQuality(id.asInstanceOf[Long]))}.print()

        ssc.start()

        val Array(train, test) = d.randomSplit(Array(0.8, 0.2))

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
