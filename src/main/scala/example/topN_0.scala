package example
import org.apache.spark.{SparkConf, SparkContext}

object topN_0 {
    val sc = new SparkContext()
    val data = sc.textFile("ip_example.txt")
    data.map(line=>"""\d+\.\d+\.\d+\.\d+""".r.findAllIn(line).mkString)
      .filter(_!="")
      .map(word=>(word,1))
      .reduceByKey(_+_)
      .map(word=>(word._2,word._1))
      .sortByKey(false)
      .map(word=>(word._2,word._1))
      .take(50)
}
