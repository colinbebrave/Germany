package example

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithSGD}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}


object LogisticRegression {

    //屏蔽不必要的日志显示在终端上
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.apache.eclipse.jetty.server").setLevel(Level.OFF)


    val conf = new SparkConf()
      .setMaster("local[4]")
      .setAppName(this.getClass().getSimpleName().filter(!_.equals('$')))

    val sc = new SparkContext(conf)

    var logisticRegression = new LogisticRegression

    //一元逻辑回归数据集
    val LR1_PATH = "file\\data\\mllib\\input\\regression\\logisticRegression1.data"
    //多元逻辑回归数据集
    val LR2_PATH = "file\\data\\mllib\\input\\regression\\sample_libsvm_data.txt"

    val data = sc.textFile(LR1_PATH)
    val svmData = MLUtils.loadLibSVMFile(sc, LR2_PATH)

    //分割数据集
    val splits = svmData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val parsedData_SVM = splits(0)
    val parsedTest_SVM = splits(1)


    //转化数据格式
    val parsedData = data.map { line =>
        val parts = line.split('|')
        LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }.cache()

    //建立模型
    val model = LogisticRegressionWithSGD.train(parsedData, 50)
    val svmModel = LogisticRegressionWithSGD.train(parsedData_SVM, 50)


    //创建测试值
    val target = Vectors.dense(-1)

    //根据模型计算测试值结果
    val predict = model.predict(target)

    //计算多元逻辑回归的测试值，并存储测试和预测值
    val predict_svm = logisticRegression.predictAndLabels(parsedTest_SVM, svmModel)

    //创建验证类
    val metrics = new MulticlassMetrics(predict_svm)

    //计算验证值
    val precision = metrics.precision

    def main(args: Array[String]) {
        println("一元逻辑回归:")
        parsedData.foreach(println)
        //打印权重
        println("权重: " + model.weights)
        println(predict)
        println(model.predict(Vectors.dense(10)))

        println("*************************************************************")

        println("多元逻辑回归:")
        println("svmData记录数：" + svmData.count())
        println("parsedData_SVM：" + parsedData_SVM.count())
        println("parsedTest_SVM：" + parsedTest_SVM.count())
        println("Precision = " + precision) //打印验证值
        predict_svm.take(10).foreach(println)
        println("权重: " + svmModel.weights)
        println("weights 个数是: " + svmModel.weights.size)
        //打印weight不为0个数
        println("weights不为0的个数是: " + model.weights.toArray.filter(_ != 0).size)
        sc.stop()
    }

}

class LogisticRegression {

    /**
      *
      * @param data  svmData
      * @param model LogisticRegressionModel
      * @return
      */
    def predictAndLabels(
                          data: RDD[LabeledPoint],
                          model: LogisticRegressionModel):RDD[(Double, Double)]= {
        val parsedData = data.map {
            point =>
                val prediction = model.predict(point.features)
                (point.label, prediction)
        }
        parsedData
    }
}
