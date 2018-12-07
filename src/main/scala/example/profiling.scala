//package example
//
//import org.apache.spark.ml.Pipeline
//import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
//import org.apache.spark.sql.SparkSession
//import org.apache.spark.ml.clustering.LDA
//import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier, NaiveBayes}
//import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator}
//import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
//
//object profiling {
//    val spark = SparkSession
//      .builder
//      .master("yarn-client")
//      .appName("Ctr Prediction Model Training")
//      .getOrCreate()
//    import spark.implicits._
//    val sentenceData = spark.createDataFrame(Seq(
//        (0, "Hi I heard about Spark"),
//        (0, "I wish Java could use case classes"),
//        (1, "Logistic regression models are neat")
//    )).toDF("label", "sentence")
//
//    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
//    val wordsData = tokenizer.transform(sentenceData)
//    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)
//    val featurizedData = hashingTF.transform(wordsData)
//
//    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
//    val idfModel = idf.fit(featurizedData)
//    val rescaledData = idfModel.transform(featurizedData)
//    rescaledData.select("features", "label").take(3).foreach(println)
//    /*输出结果为：
//    [(20,[0,5,9,17],[0.6931471805599453,0.6931471805599453,0.28768207245178085,1.3862943611198906]),0]
//    [(20,[2,7,9,13,15],[0.6931471805599453,0.6931471805599453,0.8630462173553426,0.28768207245178085,0.28768207245178085]),0]
//    [(20,[4,6,13,15,18],[0.6931471805599453,0.6931471805599453,0.28768207245178085,0.28768207245178085,0.6931471805599453]),1]
//    */
//
//    val train = spark.sparkContext.textFile("hdfs://cdh01:8020//user/data/sogou2/JBtrain", 400)
//    val test = spark.sparkContext.textFile("hdfs://cdh01:8020//user/data/sogou2/JBtest", 400)
//    val same = spark.sparkContext.textFile("hdfs://cdh01:8020//user/data/sogou2/same", 400)
//    same.filter { x => !x.contains('=') }.count()
//    val sameWord = same.map { line =>
//        val valuekey = line.split('=')
//        (valuekey(1), valuekey(0))
//    }.collect()
//    val broadcastVar = spark.sparkContext.broadcast(sameWord)
//    val diffTrain = train.map { line =>
//        val broad = broadcastVar.value
//        val regex = """^\d+$""".r
//        val temp = line.split("\t")
//        val wordArray = temp(4).split(",")
//        var str = ""
//        for (word <- wordArray) {
//            val keyVal = broad.filter(line => line._1.equals(word))
//            if (keyVal.length > 0) {
//                val oneKeyVal = keyVal(0)
//                str = str + "#" + oneKeyVal._2
//            } else if (regex.findFirstMatchIn(word) == None) {
//                str = str + "#" + word
//            }
//        }
//        (temp(0), temp(1), temp(2), temp(3), str)
//    }
//    diffTrain.toDF().coalesce(1).write.csv("hdfs://cdh01:8020//user/data/sogou2/ReplaceJBtrain")
//
//    val diffTest = test.map { line =>
//        val broad = broadcastVar.value
//        val regex = """^\d+$""".r
//        val temp = line.split("\t")
//        val wordArray = temp(1).split(",")
//        var str = ""
//        for (word <- wordArray) {
//            val keyVal = broad.filter(line => line._1.equals(word))
//            if (keyVal.length > 0) {
//                val oneKeyVal = keyVal(0)
//                str = str + "#" + oneKeyVal._2
//            } else if (regex.findFirstMatchIn(word) == None) {
//                str = str + "#" + word
//            }
//        }
//        (temp(0), str)
//    }
//    diffTest.toDF().coalesce(1).write.csv("hdfs://cdh01:8020//user/data/sogou2/ReplaceJBtest")
//
//
//    //模型评价:logLikelihood，logPerplexity。logLikelihood越大越好，logPerplexity越小越好
//    for(i<-Array(5,10,20,40,60,120,200,500)){
//        val lda=new LDA()
//          .setK(3)
//          .setTopicConcentration(3)
//          .setDocConcentration(3)
//          .setOptimizer("online")
//          .setCheckpointInterval(10)
//          .setMaxIter(i)
//        val model=lda.fit(train)
//
//        val ll = model.logLikelihood(dataset_lpa)
//        val lp = model.logPerplexity(dataset_lpa)
//
//        println(s"$i $ll")
//        println(s"$i $lp")
//    }
//
//    val topicsProb=model.transform(dataset_lpa)
//    topicsProb.select("label", "topicDistribution")show(false)
//    /**
//    +-----+--------------------------------------------------------------+
//        |label|topicDistribution                                             |
//        +-----+--------------------------------------------------------------+
//        |0.0  |[0.523730754859981,0.006564444943344147,0.46970480019667477]  |
//        |1.0  |[0.7825074858166653,0.011001204994496623,0.206491309188838]   |
//        |2.0  |[0.2085069748527087,0.005698459472719417,0.785794565674572]   |
//        ...
//
//      */
//
//
//    val lr = new MultilayerPerceptronClassifier()
//      .setMaxIter(51)
//      .setLayers(Array[Int](vector_len, 6, 5, classfiy_num))
//      .setSeed(1234L).setFeaturesCol("features")
//      .setLabelCol("label")
//      .setPredictionCol("prediction")
//
//    /**
//      * 参数调优
//      *
//      * 1 交叉验证法
//      * Spark Mllib 中实现的是留一法交叉验证法。留一法交叉验证法的思想是：将原来的训练集有N个数据集，将每一个数据集作为测试集，
//      * 其它N-1个数据集作为训练集。这样得到N个分类器，N个测试结果。用这N个结果的平均值来衡量模型的性能
//      *
//      *
//      * 参数介绍：
//      *
//      * Estimator：即所要进行评估的模型
//      * Evaluator：对模型评估器，可以是二分类的BinaryClassificationEvaluator 或是 多分类的  MulticlassClassificationEvaluator
//      * EstimatorParamMaps：模型参数表。可以由ParamGridBuilder调用addGrid方法对模型的某个参数设置一组需要验证的值，再调用build()返回。
//      * NumFolds：即所要划分的数据集的数量N
//      */
//    val pipeline = new Pipeline()
//      .setStages(Array(tokenizer, hashingTF, lr))
//
//    val paramGrid = new ParamGridBuilder()
//      .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
//      .addGrid(lr.regParam, Array(0.1, 0.01))
//      .build()
//
//    val cv = new CrossValidator()
//      .setEstimator(pipeline)
//      .setEvaluator(new BinaryClassificationEvaluator)
//      .setEstimatorParamMaps(paramGrid)
//      .setNumFolds(2)
//
//    val cvModel = cv.fit(train)
//
//    cvModel.transform(test)
//
//
//    /**
//      * 2 划分训练集验证法
//      * 划分训练集验证法的思想比较简单，我们将训练集按 m：1 - m 的比例划分成两个部分，第1部分作为新的训练集，第2部分作为验证集：
//      */
//
//    val trainValidationSplit = new TrainValidationSplit()
//      .setEstimator(lr)
//      .setEvaluator(new RegressionEvaluator)
//      .setEstimatorParamMaps(paramGrid)
//      .setTrainRatio(0.8)
//
//    val evaluations =
//        for (
//            smooth <- 1 to 100
//        ) yield {
//            val lr = new NaiveBayes().setSmoothing(smooth.toDouble / 100.0).setFeaturesCol("features")
//            val pipeline = new Pipeline().setStages(Array(lr))
//            var sumA = 0.0
//            for (cnt <- 1 to 3) {
//                val Array(trainData, testData) = dataIDF.randomSplit(Array(0.7, 0.3))
//                trainData.cache()
//                testData.cache()
//                val model = pipeline.fit(trainData)
//                val predictions = model.transform(testData)
//                val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
//                val accuracy = evaluator.evaluate(predictions)
//                sumA = sumA + accuracy
//            }
//            val allAccuracy = sumA / 3.0
//            println(((smooth), allAccuracy))
//            ((smooth), allAccuracy)
//        }
//    evaluations.sortBy(_._2).reverse.foreach(println)
//
//
//}
