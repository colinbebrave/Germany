package example
//https://blog.csdn.net/luofazha2012/article/details/80636858
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession

import scala.collection.SortedMap

object topN_2 {
    def main(args: Array[String]): Unit = {
        val num: Int = args(0).toInt
        val path: String = args(1)
        val config: SparkConf = new SparkConf().setMaster("local").setAppName("SparkNonUniqueTopN")
        val sparkContext: SparkContext = SparkSession.builder().config(config).getOrCreate().sparkContext
        val topN: Broadcast[Int] = sparkContext.broadcast(num)

        val rdd: RDD[String] = sparkContext.textFile(path)
        val kv: RDD[(String, Int)] = rdd.map(line => {
            val tokens = line.split(" ")
            (tokens(0), tokens(1).toInt)
        })

        /** convert the nonunique key to unique key
          */
        val uniqueKeys: RDD[(String, Int)] = kv.reduceByKey(_+_)
        val partitions: RDD[(Int, String)] = uniqueKeys.mapPartitions(itr => {
            var sortedMap = SortedMap.empty[Int, String]
            itr.foreach {tuple => {
                sortedMap += tuple.swap
                if (sortedMap.size > topN.value) {
                    sortedMap = sortedMap.takeRight(topN.value)
                }
            }}
            sortedMap.takeRight(topN.value).toIterator
        })

        val alltopN = partitions.collect()
        val finaltopN = SortedMap.empty[Int, String].++:(alltopN)
        val resultUsingMapPartition = finaltopN.takeRight(topN.value)

        println("+---+---+ Result1 for TopN")
        resultUsingMapPartition.foreach {
            case (k, v) => println(s"$k \t ${v.mkString(",")}")
        }

        val createCombiner: Int => Int = (v: Int) => v
        val mergeValue: (Int, Int) => Int = (a: Int, b: Int) => a + b
        val moreConciseApproach: Array[(Int, Iterable[String])] = kv.combineByKey(createCombiner, mergeValue, mergeValue)
          .map(_.swap)
          .groupByKey()
          .sortByKey(ascending = false)
          .take(topN.value)

        println("+---+---+ Result2 for TopN")
        moreConciseApproach.foreach {
            case (k, v) => println(s"$k \t ${v.mkString(",")}")
        }
    }
}
