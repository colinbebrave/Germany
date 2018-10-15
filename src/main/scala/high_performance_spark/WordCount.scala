package high_performance_spark

import org.apache.spark.rdd._

object WordCount {
  //bad idea: uses groupByKey which count trigger shuffling
  def badIdea(rdd: RDD[String]): RDD[(String, Int)] = {
    val words = rdd.flatMap(_.split(" "))
    val wordPairs = words.map((_, 1))
    val grouped = wordPairs.groupByKey()
    val wordCounts = grouped.mapValues(_.sum)
    wordCounts
  }

  //good idea: doesn't use groupByKey
  //tag::simpleWordCount[]
  def simpleWordCount(rdd: RDD[String]): RDD[(String, Int)] = {
    val words = rdd.flatMap(_.split(" "))
    val wordPairs = words.map((_, 1))
    val wordCounts = wordPairs.reduceByKey(_+_)
    wordCounts
  }

  //wordCount but filter out the illegal tokens and stop words
  def withStopWordsFiltered(rdd: RDD[String], illegalTokens: Array[Char],
                            stopWords: Set[String]): RDD[(String, Int)] = {
    val separators = illegalTokens ++ Array[Char](' ')
    val tokens: RDD[String] = rdd.flatMap(_.split(separators).map(_.trim.toLowerCase))

    val words = tokens.filter(token => !stopWords.contains(token) && (token.length > 0))

    val wordPairs = words.map((_, 1))
    val wordCounts = wordPairs.reduceByKey(_+_)

    wordCounts
  }
}
