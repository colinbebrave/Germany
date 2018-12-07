package nlp.sam_spam_classification_multilayer_perceptron
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, Word2Vec}
import org.apache.spark.sql.SparkSession

object spam_classification {

    def main(args: Array[String]): Unit = {
        val sc = new SparkContext()
        val spark = SparkSession
          .builder
          .master("local[*]")
          .appName("training Naive Bayes model")
          .getOrCreate()

        //val msgDF = spark.read.option("inferschema", "true").option("delimiter", " ").csv("/Users/wangqi/Documents/Java/Germany/src/main/scala/nlp/sam_spam_classification_multilayer_perceptron/")

        /**
          * 1读取文件并分词
          */

        val msgRDD = sc.textFile("/Users/wangqi/Documents/Java/Germany/src/main/scala/nlp/sam_spam_classification_multilayer_perceptron/SMSSpamCollection", minPartitions = 10)

        val parsedRDD = msgRDD.map(_.split("\t")).map ( row => {
              val label = row(0)
              val text = row(1).split(",").flatMap(word => word.trim.split(" "))
            (label, text)
        })

        val msgDF = spark.createDataFrame(parsedRDD).toDF("label", "message")


        /**
          * 2将标签转化为索引
          */

        val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(msgDF)

        /**
          * 3创建word vector, 维度为100
          */

        val VECTOR_SIZE = 100
        val word2Vec = new Word2Vec().setInputCol("message").setOutputCol("features").setVectorSize(VECTOR_SIZE).setMinCount(1)

        /**
          * 4创建多层感知器
          * 输入层为VECTOR_SIZE个，
          * 中间层分别是6， 5个神经元
          * 输出层是2个
          */

        val layers = Array[Int](VECTOR_SIZE, 6, 5, 2)
        val mlpc = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(512).setSeed(1234L).setMaxIter(128).setFeaturesCol("features").setLabelCol("indexedLabel").setPredictionCol("prediction")

        /**
          * 5将索引转换为原有标签
          */

        val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

        /**
          * 6数据集分割
          */

        val Array(trainingData, testingData) = msgDF.randomSplit(Array(0.8, 0.2))

        /**
          * 7创建pipeline并训练数据
          */

        val pipeline = new Pipeline().setStages(Array(labelIndexer, word2Vec, mlpc, labelConverter))
        val model = pipeline.fit(trainingData)

        val predictionResultDF = model.transform(testingData)

        predictionResultDF.printSchema()
        predictionResultDF.select("message", "label", "predictedLabel").show(30, false)

        /**
          * 8评估模型效果
          */

        val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("weightedPrecision")
        val predictionAccuracy = evaluator.evaluate(predictionResultDF)
        println("Testing Accuracy is %2.4f".format(predictionAccuracy * 100) + "%")







} //for main function

} // for the object definition
