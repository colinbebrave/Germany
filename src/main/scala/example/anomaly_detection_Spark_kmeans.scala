package example

//https://github.com/keiraqz/anomaly-detection

import java.io._

import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer

object anomaly_detection_Spark_kmeans {
    def main(args: Array[String]): Unit = {
        val sparkConf = new SparkConf().setAppName("AnomalyDetection")
        val sc = new SparkContext(sparkConf)
        val normalizedData = loadData(sc)
        val model = trainModel(normalizedData)
        val file = new File("/Users/wangqi/Desktop/trainOutput.txt")
        val bw = new BufferedWriter(new FileWriter(file))
        val centroid = model.clusterCenters(0).toString //save centroid to file
        bw.write(centroid + ",")

        //decide threshold for anomalies
        val distances = normalizedData.map(d => distToCentroid(d, model))
        val threshold = distances.top(2000).last //set the last of the furtherest 2000 data points as the threshold
        bw.write(threshold.toString) //last item is the threshold
        bw.close()
    }

    /**
      * load data from file, parse the data and normalize the data
      */
    def loadData(sc: SparkContext): RDD[Vector] = {
        val rawData = sc.textFile("/Users/wangqi/Desktop/train.csv", 120)
        //parse data file
        val dataAndLabel = rawData.map { line =>
            val buffer = ArrayBuffer[String]()
            buffer.appendAll(line.split(","))
            buffer.remove(1, 3) //remove categorical attributes
            val label = buffer.remove(buffer.length - 1)
            val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
            (vector, label)
        }

        val data = dataAndLabel.map(_._1).cache()
        val normalizedData = normalization(data)
        normalizedData
    }

    /**
      * normalization function, normalize the training data
      */
    def normalization(data: RDD[Vector]): RDD[Vector] = {
        val dataArray = data.map(_.toArray)
        val numCols = dataArray.first().length
        val n = dataArray.count()
        val sums = dataArray.reduce((a, b) => a.zip(b).map(t => t._1 + t._2))
        val sumSquares = dataArray.fold(new Array[Double](numCols)) (
            (a, b) => a.zip(b).map(t => t._1 + t._2 * t._2)
        )
        val stdevs = sumSquares.zip(sums).map{
            case (sumSq, sum) => math.sqrt(n * sumSq - sum * sum) / n
        }
        val means = sums.map(_/n)

        def normalize(v: Vector): Vector = {
            val normed = (v.toArray, means, stdevs).zipped.map {
                case (value, mean, 0) => (value - mean) / 1 //if stdev is 0
                case (value, mean, stdev) => (value - mean) / stdev
            }
            Vectors.dense(normed)
        }

        val normalizedData = data.map(normalize(_)) //do normalization
        normalizedData
    }

    /**
      * train a KMean model using normalized data
      */
    def trainModel(normalizedData: RDD[Vector]): KMeansModel = {
        val kmeans = new KMeans()
        kmeans.setK(1) //find that one center
        kmeans.setRuns(10)
        val model = kmeans.run(normalizedData)
        model
    }

    /**
      * calculate distance between data point to centroid
      */
    def distToCentroid(vector: Vector, model: KMeansModel): Double = {
        val centroid = model.clusterCenters(model.predict(vector))//if more than 1 center
        Vectors.sqdist(vector, centroid)
    }
}
