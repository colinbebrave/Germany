package example
//https://blog.csdn.net/luofazha2012/article/details/80636858

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object topN_3_within_group {
    def main(args: Array[String]): Unit = {
        val num = args(0).toInt
        val path: String = args(1)
        val config: SparkConf = new SparkConf().setMaster("local").setAppName("SparkGroupTopN")
        val sparkContext: SparkContext = SparkSession.builder().config(config).getOrCreate().sparkContext

        val rdd: RDD[String] = sparkContext.textFile(path)
        //filter blank lines, convert rdd into PairRDD
        val mapredRDD: RDD[(String, Int)] = rdd.filter(line => line.length > 0)
          .map(line => line.split(" "))
          .map(arr => (arr(0).trim, arr(1).trim.toInt))

        //cache RDD
        mapredRDD.cache()
        val topN: Broadcast[Int] = sparkContext.broadcast(num)


        /**
          * 1.get TopN using groupByKey
          * Cons:
          * 1) data with same key will be loaded into memory which would result in memory error
          * 2) it is inefficient
          */
        val topNResult1: RDD[(String, Seq[Int])] = mapredRDD.groupByKey().map(
            tuple2 => {
                //get topN in values
                val topn = tuple2._2.toList.sorted.takeRight(topN.value).reverse
                (tuple2._1, topn)
            })
        println("+---+---+ result for topN using groupByKey: ")
        println(topNResult1.collect().mkString("\n"))

        /**
          * 2.two-stage aggregation.
          * firstly aggregate within groups by random numbers to get local topN
          * then, get global topN
          */
        val topNResult2: RDD[(String, List[Int])] = mapredRDD.mapPartitions(iterator => {
            iterator.map(tuple2 => {
                ((Random.nextInt(10), tuple2._1), tuple2._2)
            })
        }).groupByKey().flatMap({
            case ((_, key), values) =>
                values.toList.sorted.takeRight(topN.value).map(value => (key, value))
        }).groupByKey().map(tuple2 => {
            val topn = tuple2._2.toList.sorted.takeRight(topN.value).reverse
            (tuple2._1, topn)
        })
        println("+---+---+ result for topN using two-stage aggregation: ")
        println(topNResult2.collect().mkString("\n"))

        /**
          * 3.using aggregateByKey
          */
        val topNResult3: RDD[(String, List[Int])] = mapredRDD.aggregateByKey(ArrayBuffer[Int]())(
            (u, v) => {
                u += v
                u.sorted.takeRight(topN.value)
            },
            (u1, u2) => {
                u1 ++= u2
                u1.sorted.takeRight(topN.value)
            }
        ).map(tuple2 => (tuple2._1, tuple2._2.toList.reverse))
        println("+---+---+ result for topN using two-stage aggregation: ")
        println(topNResult3.collect().mkString("\n"))
    }
}
