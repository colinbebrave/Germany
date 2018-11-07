package Scala_and_Spark_for_Big_Data_Analysis

import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{FloatType, StructType}

//to convert string to integer in scala spark dataframe
import org.apache.spark.sql.types.IntegerType

//for keywords: when and col, must import the following
import org.apache.spark.sql.functions._


object chapter11_LR {

    case class Cancer(cancer_class: Double,
                      thickness: Double,
                      size: Double,
                      shape: Double,
                      madh: Double,
                      epsize: Double,
                      bnuc: Double,
                      bchrom: Double,
                      nNuc: Double,
                      mit: Double)

    def parseRDD(rdd: RDD[String]): RDD[Array[Double]] = {
        rdd.map(_.split(",")).filter(_(6) != "?").map(_.drop(1)).map(_.map(_.toDouble))
    }

    def parseCancer(line: Array[Double]): Cancer = {
        Cancer(if (line(9) == 4.0) 1 else 0, line(0), line(1), line(2), line(3), line(4), line(5), line(6), line(7), line(8))
    }

    /**
      * 1.load and parse the data
      */
    val spark = SparkSession.builder().master("local[*]").getOrCreate()
    import spark.implicits._

    val rdd = spark.sparkContext.textFile("wbcd.csv")
    val cancerRDD = parseRDD(rdd)

    /**
      * 2.convert RDD to DataFrame for the ML Pipeline
      */
    val schema = new StructType()
      .add("thickness", FloatType)
      .add("size", FloatType)
      .add("shape", FloatType)
      .add("madh", FloatType)
      .add("epsize", FloatType)
      .add("bnuc", FloatType)
      .add("bchrom", FloatType)
      .add("nNuc", FloatType)
      .add("mit", FloatType)
      .add("cancer_class", FloatType)

    val gs = spark.read
      .option("header", "true")
      .option("inferschema", "true")
      .option("sep", ",")
      .csv("Desktop/wbc.txt")

    val gss = gs.drop(gs.col("id"))

    //transform cancer_class column, 2 -> 0, 4 -> 1
    val data = gss.withColumn("cancer_class", when(col("cancer_class") ===2, 0).otherwise(1))

    val modify_class = udf{(cancer_class: Int) => if (cancer_class == 2) 0 else 1}

    val data1 = gss.withColumn("cancer_class", modify_class(gss.col("cancer_class")))
    /*
      * "thickness", "size", "shape", "madh", "epsize", "bnuc", "bchrom", "nNuc", "mit", "cancer_class"
      */

    data1.cache()
    //found that elements if column bnuc are strings, convert them to integers firstly
    //to show the way to drop column and rename a column in spark DataFrame we first use this boring method
    val data2 = data1.withColumn("bnuc_temp", data1("bnuc").cast(IntegerType)).drop("bnuc").withColumnRenamed("bnuc_temp", "bnuc")
    /*
      * actually the above action can be done much simplified
      * val data2 = data1.withColumn("bnuc", data1("bnuc").cast(IntegerType))
      * we could assign the same name to the new column for which we do not need to drop the original column and rename the new generated column
      */
    data1.unpersist()


    /**
      * 3.Feature extraction and transformation
      */
    val featureCols = Array("thickness", "size", "shape", "madh", "epsize", "bnuc", "bchrom", "nNuc", "mit")

    /*
      * let us assemble feature columns into a feature vector
      */

    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val data3 = assembler.transform(data2)

    /*
      * let us encode the label
      */
    val labelIndexer = new StringIndexer().setInputCol("cancer_class").setOutputCol("label")
    val data4 = labelIndexer.fit(data3).transform(data3)

    /**
      * 4.creating test and training set
      */

    val splitSeed = 777
    val Array(train, test) = data4.randomSplit(Array(0.7, 0.3), splitSeed)

    /**
      * 5.fitting the training set
      */

    val lr = new LogisticRegression().setMaxIter(50).setRegParam(0.01).setElasticNetParam(0.01)
    val model = lr.fit(train)

    /**
      * 6.getting the raw prediction, probability and prediction for the test set
      */

    val predictions = model.transform(train)
    predictions.show()

    /**
      * 7.generating objective history of training
      */
    val trainingSummary = model.summary
    val objectiveHistory = trainingSummary.objectiveHistory
    objectiveHistory.foreach(loss => println(loss))
    //the above method will print the training loss of each iteration, the loss gradually reduces in later iteration

    /**
      * 8.evaluate the model
      */
    val binarySummary = trainingSummary.asInstanceOf[BinaryLogisticRegressionSummary]
    val roc = binarySummary.roc
    roc.show()
    println("Area Under ROC: " + binarySummary.areaUnderROC)

    /**
      * 9.manually compute metrics like TPR, FPR, FNR, etc
      */

    val lp = predictions.select("label", "prediction")

    val counttotal = predictions.count()
    val correct = lp.filter($"label" === $"prediction").count()
    val wrong = lp.filter(not($"label" ===$"prediction")).count()
    val trueP = lp.filter($"prediction" === 0.0).filter($"label" === $"prediction").count()
    val ratiowrong = wrong.toDouble / counttotal.toDouble

//
//    val fMeasure = binarySummary.fMeasureByThreshold
//    val fm = fMeasure.col("F-Measure")
//    val maxFMeasure = fMeasure.select(max("F-Measure")).head().getDouble(0)
//    val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure).select("threshold").head().getDouble(0) model.setThreshold(bestThreshold)
//
//    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label") val accuracy = evaluator.evaluate(predictions) println("Accuracy: " + accuracy)
}


























