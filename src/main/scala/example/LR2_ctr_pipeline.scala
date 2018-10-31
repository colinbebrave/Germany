package example

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession

object LR2_ctr_pipeline {
    val spark = SparkSession.builder()
      .appName("Ctr Prediction Model Training")
      .getOrCreate()

    //for testing dataset, randomly assign value between 1-5
    val ctrPredictTrain = spark.read.format("com.databricks.spark.csv")
      .option("delimiter", ",")
      .option("header", true)
      .load("train.csv")
    //ctrPredictTrain.printSchema()

    ctrPredictTrain.createOrReplaceTempView("CtrPredTrain")

    val ctrPrediction = spark.sql("SELECT * FROM CtrPredTrain")

    //check if there are duplicated ad ID
    //val countUniqueId = ctrPrediction.selectExpr("*").groupBy("id").count().sort(desc("count"))
    //countUniqueId.take(5)

    /***
      * Spark Pipeline for transforming the data
      * Include one hot encoding for categorical variables
      * Combine all features into one feature vector column
      */

    //transform categorical features into indexed feature
    val stringFeatures = Array("site_domain", "site_category", "app_domain", "app_category", "device_model")
    val catCol = Array("C1", "banner_pos", "device_type", "devive_conn_type", "C14", "C15", "C16", "C17", "C17", "C18", "C19", "C20", "C21")
    val stringCatFeatures = stringFeatures ++ catCol
    val catFeaturesIndexer = stringCatFeatures.map(
        cname => new StringIndexer()
          .setInputCol(cname)
          .setOutputCol(s"${cname}_index")
    )

    val indexPipeline = new Pipeline().setStages(catFeaturesIndexer)
    val model = indexPipeline.fit(ctrPrediction)
    val indexedDF = model.transform(ctrPrediction)

    //one how encoding for categorical features
    val indexedCols = indexedDF.columns.filter(x => x contains "index")
    val indexedFeatureEncoder = indexedCols.map(
        indexed_cname => new OneHotEncoder()
          .setInputCol(indexed_cname)
          .setOutputCol(s"${indexed_cname}_vec")
    )

    val encodedPipeline = indexPipeline.setStages(indexedFeatureEncoder)
    val encodeModel = encodedPipeline.fit(indexedDF)
    val encodeDf = encodeModel.transform(indexedDF)

    //ad id is not a feature, neither is the output "click"
    //also only keep encoded feature and ignore original cat features and their index
    val nonFeatureCol = Array("id", "click", "site_id", "app_id", "device_id")
    val featureCol = encodeDf.columns.filter(x => x contains "_vec")

    //conbine all feature columns into a big column
    val assembler = new VectorAssembler().setInputCols(featureCol).setOutputCol("features")
    val encodedTrainingSet = assembler.transform(encodeDf)

    //convert click to double
    val finalTrainSet = encodedTrainingSet.selectExpr("*", "double(click) as click_output")

    /***
      * split data and ready for training and testing
      */
    //split the data for training
    val Array(train, test) = finalTrainSet.randomSplit(Array(0.7, 0.3))

    train.cache()
    test.cache()

    println(train.count())
    println(test.count())

}
