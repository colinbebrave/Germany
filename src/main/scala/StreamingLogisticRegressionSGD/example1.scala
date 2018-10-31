package StreamingLogisticRegressionSGD
import org.apache.spark.mllib.classification.StreamingLogisticRegressionWithSGD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.doubleRDDToDoubleRDDFunctions
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.{SparkConf, SparkContext}

object LogisticRegressionApp {
    def main(args: Array[String]): Unit = {
        if (args.length != 4) {
            System.err.println(
                "Usage: LogisticRegressionAPP <appname> <batchInterval> <hostname> <port>")
            System.exit(1)
        }
        val Seq(appName, batchInterval, hostname, port) = args.toSeq
        val conf = new SparkConf()
          .setAppName(appName)
          .setJars(SparkContext.jarOfClass(this.getClass).toSeq)

        val ssc = new StreamingContext(conf, Seconds(batchInterval.toInt))

        val substream = ssc.socketTextStream(hostname, port.toInt)
          .filter(!_.contains("Nan"))
          .map(_.split(" "))
          .filter(f => f(1) != "0")

        val datastream = substream.map(f => Array(f(1).toDouble, f(2).toDouble, f(4).toDouble, f(5).toDouble, f(6).toDouble))

        val walkingOrRunning = datastream.filter(f => f(0) == 4.0 || f(0) == 5.0).map(f => LabeledPoint(f(0), Vectors.dense(f.slice(1, 5))))

        val test = walkingOrRunning.transform(rdd => rdd.randomSplit(Array(0.3, 0.7))(0))
        val train = walkingOrRunning.transformWith(test, (r1: RDD[LabeledPoint], r2: RDD[LabeledPoint]) => r1.subtract(r2)).cache()

        val model = new StreamingLogisticRegressionWithSGD()
          .setInitialWeights(Vectors.zeros(4))
          .setStepSize(0.0001)
          .setNumIterations(1)

        model.trainOn(train)
        model.predictOnValues(test.map(v => (v.label, v.features))).foreachRDD(rdd => println("MSE: %f".format(
            rdd.map(v => math.pow((v._1 - v._2), 2)).mean())))

        ssc.start()
        ssc.awaitTermination()
    }
}
