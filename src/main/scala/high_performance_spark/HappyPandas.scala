package high_performance_spark

import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, DataFrame, SparkSession, Row}
import org.apache.spark.sql.catalyst.expressions.aggregate._
import org.apache.spark.sql.expressions._
import org.apache.spark.sql.functions._

//tag::legacySparkSQLImports[]
import org.apache.spark.sql.SQLContext
//end::legacySparkSQLImports[]


object HappyPandas {

    def sparkSession(): SparkSession = {
        //tag::createSparkSession[]
        val session = SparkSession.builder()
          .enableHiveSupport()
          .getOrCreate()
        //Import the implicits, unlike in core Spark the implicits are defined on the context
        import session.implicits._
        //end::createSparkSession[]
        session
    }

    //create SQLContext with an existing SparkContext
    def sqlContext(sc: SparkContext): SQLContext = {
        //tag::createSQLContext[]
        val sqlContext = new SQLContext(sc)
        //Import the implicits, unlike in core Spark the implicits are defined on the context
        import sqlContext.implicits._
        //end::createSparkSession[]
        sqlContext
    }

    //illustrate loading some JSON data
    def loadDataSimple(sc: SparkContext, session: SparkSession, path: String): DataFrame = {
        //tag::loadPandaJSONSimple[]
        val df1 = session.read.json(path)
        //end::loadPandaJSONSimple[]

        //tag::loadPandaJSONComplex[]
        val df2 = session.read.format("json")
          .option("samplingRatio", "1.0").load(path)
        //end::loadPandaJSONComplex[]

        val jsonRDD = sc.textFile(path)
        //tag::loadPandaJsonRDD[]
        val df3 = session.read.json(jsonRDD)
        //end::loadPandaJsonRDD[]

        df1
    }

    def jsonLoadFromRDD(session: SparkSession, input: RDD[String]): DataFrame = {
        //tag::loadPandaJSONRDD[]
        val rdd: RDD[String] = input.filter(_.contains("panda"))
        val df = session.read.json(rdd)
        //end::loadPandaJSONRDD[]
        df
    }

    /**
      * @param place name of place
      * @param pandaType type of pandas in this place
      * @param happyPandas number of happy pandas in this place
      * @param totalPandas total number of pandas in this place
      */

    case class PandaInfo(
                        place: String,
                        pandaType: String,
                        happyPandas: Integer,
                        totalPandas: Integer
                        )

    /**
      * gets the percentage of happy pandas per place
      *
      * @param pandaInfo the input DataFrame
      * @return Returns DataFrame of (place, percentage of happy pandas)
      */

    def happyPandasPercentage(pandaInfo: DataFrame): DataFrame = {
        pandaInfo.select(
            pandaInfo("place"),
            (pandaInfo("happyPandas") / pandaInfo("totalPandas")).as("percentHappy")
        )
    }

    //tag::encodePandaType[]
    /**
      * Encodes pandaType to Integer values instead of String values
      *
      * @param pandainfo the input DataFrame
      * @return Returns a DataFrame of pandaId and integer value for pandaType
      */
    def encodePandaType(pandaInfo: DataFrame): DataFrame = {
        pandaInfo.select(pandaInfo("id"),
            (when(pandaInfo("pt") === "giant", 0).
              when(pandaInfo("pt") === "red", 1).
              otherwise(2)).as("encodedType")
        )
    }
    //end::encodePandaType[]

    /**
      * gets places with happy pandas more than minHappinessBound
      */
    def minHappyPandas(pandaInfo: DataFrame, minHappyPandas: Int): DataFrame = {
        //tag::selectExplode[]
        pandaInfo.filter(pandaInfo("happyPandas") >= minHappyPandas)
    }

    /**
      * extra the panda info from panda places and compute the squishness of the panda
      */

    def sadPandas(pandaInfo: DataFrame): DataFrame = {
        //tag::simpleFilter[]
        pandaInfo.filter(pandaInfo("happy") =!= true)
        //end::simpleFilter[]
    }

    /**
      * find pandas that are fuzzier than squishy
      */

    def happyFuzzyPandas(pandaInfo: DataFrame): DataFrame = {
        //tag::complexFilter[]
        pandaInfo.filter(
            pandaInfo("happy").and(pandaInfo("attributes")(0) > pandaInfo("attributes")(1))
        )
    }

    /**
      * gets places that contains happy pandas more than unhappy pandas
      */

    def happyPandasPlaces(pandaInfo: DataFrame): DataFrame = {
        pandaInfo.filter(pandaInfo("happyPandas") >= pandaInfo("totalPandas") / 2)
    }

    /**
      * remove duplicate pandas by id
      */

    def removeDuplicates(pandas: DataFrame): DataFrame = {
        //tag::dropDuplicatesPandaIds[]
        pandas.dropDuplicates(List("id"))
    }

    /**
      * @param name name of panda
      * @param zip code
      * @param pandaSize size of panda in KG
      * @age age of panda
      */

    case class Pandas(name: String, zip: String, pandaSize: Integer, age: Integer)

    def describePandas(pandas: DataFrame) = {
        //tag::pandaSizeRangeVarDescribe[]
        //compute the count, mean, stddev, min, max summary stats for
        //all of the numeric fields of the provided panda infos. non-numeric fields
        //(such as string (name) or array types are skipped
        val df = pandas.describe()
        //collect the summary back locally
        println(df.collect())
        //end::pandaSizeRangeVarDescribe[]
    }

    //tag::maxPandaSizePerZip[]
    def maxPandaSizePerZip(pandas: DataFrame): DataFrame = {
        pandas.groupBy(pandas("zip")).max("pandaSize")
    }
    //end::maxPandaSizePerZip[]

    //tag::minMaxPandasSizePerZip[]
    def minMaxPandaSizePerZip(pandas: DataFrame): DataFrame = {
        pandas.groupBy(pandas("zip")).agg(min("pandasSize"), max("PandaSize"))
    }
    //end::minMaxPandasPerZip[]

    def minPandaSizeMaxAgePerZip(pandas: DataFrame): DataFrame = {
        //this query can be written in two methods

        //1
        pandas.groupBy(pandas("zip")).agg(("pandasSize", "Min"), ("age", "max"))

        //2
        pandas.groupBy(pandas("zip")).agg(Map("pandaSize" -> "min", "age" -> "max"))
    }

    //tag::complexAggPerZip[]
    def minMeanSizePerZip(pandas: DataFrame): DataFrame = {
        //compute the min and mean
        pandas.groupBy(pandas("zip")).agg(
            min(pandas("pandaSize")), mean(pandas("pandaSize"))
        )
    }
    //end::complexAggPerZio[]

    def simpleSqlExample(pandas: DataFrame): DataFrame = {
        val session = pandas.sparkSession
        //tag::pandasSQLQuery
        pandas.createOrReplaceTempView("pandas")
        val miniPandas = session.sql("select * from pandas where pandaSize < 12")
        //end::pandasSQLQuery
        miniPandas
    }

    /**
      * Orders panda by size ascending and by age descending
      * pandas will be sorted by "size" first and if two pandas have the same "size"
      * will be sorted by age
      */

    def orderPandas(pandas: DataFrame): DataFrame = {
        //tag::simpleSort[]
        pandas.orderBy(pandas("pandaSize").asc, pandas("age").desc)
        //end::simpleSort[]
    }

    def computeRelativePandaSize(pandas: DataFrame): DataFrame = {
        //tag::relativePandaSizesWindows[]
        val windowSpec = Window
          .orderBy(pandas("age"))
          .partitionBy(pandas("zip"))
          .rowsBetween(start = -10, end = 10) //can use rangeBetween for range instead
        //end::relativePandaSizeWindowp[]

        //tag::relativePandaSizesQuery[]
        val pandaRelativeSizeCol = pandas("pandaSize") - avg(pandas("pandaSize")).over(windowSpec)

        pandas.select(pandas("name"), pandas("zip"), pandas("pandaSize"), pandas("age"),
            pandaRelativeSizeCol.as("panda_relative_size"))
    }

    //Join DataFrames of Pandas and Sizes with
    def joins(df1: DataFrame, df2: DataFrame): Unit = {
        //tag::innerJoin[]
        //Inner join implicit
        df1.join(df2, df1("name") === df2("name"))
        //Inner join explicit
        df1.join(df2, df1("name") === df2("name"), "inner")
        //end::innerJoin[]

        //tag::leftouterJoin[]
        //Left outer join explicit
        df1.join(df2, df1("name") === df2("name"), "left_outer")

        //tag::rightouterJoin[]
        //Right outer join explicit
        df1.join(df2, df1("name") === df2("name"), "right_outer")
        //end::rightouterJoin[]

        //tag::leftSemiJoin[]
        //left semi join explicit
        df1.join(df2, df1("name") === df2("name"), "left_semi")
        //end::leftsemiJoin[]
    }

    /**
      * cut the lineage of a DataFrame which has too long a query plan
      */

    def cutLineage(df: DataFrame): DataFrame = {
        val sqlCtx = df.sqlContext
        //tag::cutLineage[]
        val rdd = df.rdd
        rdd.cache()
        sqlCtx.createDataFrame(rdd, df.schema)
        //end::cueLineage
    }

    //tag::self join
    //Self join
    def selfJoin(df: DataFrame): DataFrame = {
        val sqlCtx = df.sqlContext
        import sqlCtx.implicits._
        //tag::selfJoin[]
        val joined = df.as("a").join(df.as("b")).where($"a.name" === $"b.name")
        //end::selfJoin[]
        joined
    }
}
