package example
//https://blog.csdn.net/luofazha2012/article/details/80636858
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD

import scala.collection.SortedMap

object topN_1 {
    def main(args: Array[String]): Unit = {
        //N
        val num: Int = args(0).toInt
        //path to file
        val path: String = args(1)
        val config: SparkConf = new SparkConf().setMaster("local").setAppName("SparkUniqueTopN")
        //construct sparkcontext
        val sparkContext: SparkContext = SparkSession.builder().config(config).getOrCreate().sparkContext

        //broadcast variable
        val topN: Broadcast[Int] = sparkContext.broadcast(num)

        val rdd: RDD[String] = sparkContext.textFile(path)
        val pairRdd: RDD[(Int, Array[String])] = rdd.map(line => {
            val tokens: Array[String] = line.split(" ")
            (tokens(1).toInt, tokens)
        })

        val partitions: RDD[(Int, Array[String])] = pairRdd.mapPartitions(iterator => {
            var sortedMap = SortedMap.empty[Int, Array[String]]
            iterator.foreach({tuple => {
                sortedMap += tuple
                if (sortedMap.size > topN.value) {
                    sortedMap = sortedMap.takeRight(topN.value)
                }
            }})
            sortedMap.takeRight(topN.value).toIterator
        })

        val alltopN: Array[(Int, Array[String])] = partitions.collect()
        val finaltopN: SortedMap[Int, Array[String]] = SortedMap.empty[Int, Array[String]].++:(alltopN)
        val resultUsingPartition: SortedMap[Int, Array[String]] = finaltopN.takeRight(topN.value)

        print("+---+---+ Result1 for TopN")
        resultUsingPartition.foreach {
            case (k, v) => println(s"$k \t ${v.asInstanceOf[Array[String]].mkString(",")}")
        }

        val moreConciseApproach: Array[(Int, Iterable[Array[String]])] = pairRdd.groupByKey().sortByKey(ascending = false).take(topN.value)
        print("+---+---+ Result2 for TopN")
        moreConciseApproach.foreach {
            case (k, v) => println(s"$k \t ${v.flatten.mkString(",")}")
        }
    }
}
