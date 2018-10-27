package Scala_and_Spark_for_Big_Data_Analysis
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.feature.{CountVectorizer, StopWordsRemover, Tokenizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

object chapter15_text_classification {
    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("feature extraction")
      .getOrCreate()
    import spark.implicits._

    /**
      * 1.load the input text data
      */
    val inputText = spark.sparkContext.textFile("Documents/Java/Germany/src/main/scala/Scala_and_Spark_for_Big_Data_Analysis/Sentiment_Analysis_Dataset10k.csv")

    /**
      * 2.convert the input lines to a dataframe
      */
    val sentenceDF = inputText.map(x => (x.split(",")(0),
    x.split(",")(1),
    x.split(",")(2))).toDF("id", "label", "sentence")

    /**
      * 3.transform the data into words using a Tokenizer
      */
    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsDF = tokenizer.transform(sentenceDF)
    wordsDF.show(5, false)

    /**
      * 4.remove stop words and create a new DataFrame with the filtered words
      */
    val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filteredWords")
    val noStopWordsDF = remover.transform(wordsDF)
    noStopWordsDF.show(5, false)

    /**
      * 5.create a feature vector from the filtered words
      */
    val countVectorizer = new CountVectorizer().setInputCol("filteredWords").setOutputCol("features")
    val countVectorizerModel = countVectorizer.fit(noStopWordsDF)
    val countVectorizerDF = countVectorizerModel.transform(noStopWordsDF)

    /**
      * 6.create the inputData DataFrame with just a label and the features
      */
    val inputData = countVectorizerDF.select("label", "features").withColumn("label", col("label").cast("double"))

    /**
      * 7.split the data using a random split into 80% training and 20% testing datasets
      */
    val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2))

    /**
      * 8.create a logistic regression model
      */
    val lr = new LogisticRegression()
    var lrModel = lr.fit(train)
    lrModel.coefficients
    lrModel.intercept

    /**
      * 9.examine the model summary especially areaUnderROC, which should be >0.90 for a good model
      */

    val summary = lrModel.summary
    val bSummary = summary.asInstanceOf[BinaryLogisticRegressionSummary]
    bSummary.areaUnderROC
    bSummary.roc
    bSummary.pr.show()

    /**
      * 10.transform both train and test datasets using the trained model
      */
    val training = lrModel.transform(train)
    val testing = lrModel.transform(test)
}


















