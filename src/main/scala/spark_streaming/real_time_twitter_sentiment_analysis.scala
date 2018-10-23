package spark_streaming

//https://stdatalabs.com/2016/09/running-apache-spark-on-eclipse/

import org.apache.spark.streaming._
import org.apache.spark.SparkContext._
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.twitter._
import org.apache.spark.streaming.flume._

object real_time_twitter_sentiment_analysis {

    /**
      * A spark Streaming application that receives tweets on certain keywords
      * from twitter datasource and find the popular hashtags
      * <consumerKey>                 -Twitter consumer key
      * <consumerSecret>              -Twitter consumer secret
      * <accessToken>                 -Twitter access token
      * <accessTokenSecret>           -Twitter access token secret
      * <keyword_1>                   -The keyword to filter tweets
      * <keyword_n>                   -Any number of keywords to filter tweets
      */

    val conf = new SparkConf().setMaster("local[*]").setAppName("Spark Streaming - PopularHashTags")
    val sc = new SparkContext(conf)

    def main(args: Array[String]): Unit = {
        sc.setLogLevel("WARN")

        val Array(consumerKey, consumerSecret, accessToken, accessTokenSecret) = args.take(4)
        val filters = args.takeRight(args.length - 4)

        // Set the system properties so that Twitter4j library used by twitter stream
        // can use them to generat OAuth credentials
        System.setProperty("twitter4j.oauth.consumerKey", consumerKey)
        System.setProperty("twitter4j.oauth.consumerSecret", consumerSecret)
        System.setProperty("twitter4j.oauth.accessToken", accessToken)
        System.setProperty("twitter4j.oauth.accessTokenSecret", accessTokenSecret)

        //set the spark streamingcontext to create a DStream for every 5 seconds
        val ssc = new StreamingContext(sc, Seconds(5))
        //pass the filter keywords as arguments

        val stream = TwitterUtils.createStream(ssc, None, filters)

        //split the stream on space and extract hashtags
        val hashTags = stream.flatMap(status => status.getText.split(" ").filter(_.startsWith("#")))
        val topCounts60 = hashTags.map((_, 1)).reducebyKeyAndWindow(_+_, Seconds(60))
          .map{case (topic, count) => (count, topic)}
          .transform(_.sortByKey(false))

        //print
        stream.print()

        //print popular hashtags
        topCounts60.foreachRDD(rdd => {
            val topList = rdd.take(10)
            println("\nPopular topics in last 10 seconds (%s total: ".format(rdd.count()))
            topList.foreach {case (count, tag) => println("%s (%s tweets".format(tag, count))}
        })

        ssc.start()
        ssc.awaitTermination()
    }
}
