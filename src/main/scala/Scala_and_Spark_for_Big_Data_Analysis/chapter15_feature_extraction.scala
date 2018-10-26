package Scala_and_Spark_for_Big_Data_Analysis

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{Tokenizer, RegexTokenizer, StopWordsRemover,
    NGram, HashingTF, IDF, Word2Vec, CountVectorizer}

object chapter15_feature_extraction {
    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("feature extraction")
      .getOrCreate()

    import spark.implicits._
    val lines = Seq(
        (1, "Hello there, how do you like the book so far?"),
        (2, "I am new to Machine Learning"),
        (3, "Maybe i should get some coffee before starting"),
        (4, "Coffee is best when you drink it hot"),
        (5, "Book stores have coffee too so i should go to a book store")
    )

    val sentenceDF = spark.createDataFrame(lines).toDF("id", "sentences")
    sentenceDF.show(10, false)
    /**
      * the most common techniques for feature extraction are as followss:
      * 1.Tokenizer / RegexTokenizer
      * 2.StopWordsRemover
      * 3.Binarizer
      * 4.NGrams
      * 5.HashingTF
      * 4.TF-IDF
      * 5.Word2Vec
      * 6.CountVectorizer
      */

    /**
      * 1.Tokenization
      * Tokenizer converts the input string into lowercase and then splits the string with whitespaces
      * into invididual tokens.
      * RegexTokenizer splits the string based on regular expression
      */

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsDF = tokenizer.transform(sentenceDF)
    wordsDF.show(10, false)

    val regexTokenizer = new RegexTokenizer().setInputCol("sentence").setOutputCol("regexWords").setPattern("\\W")
    val regexWordsDF = regexTokenizer.transform(sentenceDF)
    regexWordsDF.show(10, false)

    /**
      * 2.StopWordsRemover
      * StopWordsRemover removes the commonly used words with the inner dictionary
      */
    val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filteredWords")
    val noStopWordsDF = remover.transform(wordsDF)
    noStopWordsDF.select("sentence", "filteredWords").show(10, false)

    //Stop words are set by default, but can be overwritten or amended very easily
    val noHello = Array("hello") ++ remover.getStopWords
    val removerCustom = new StopWordsRemover().setInputCol("words").setOutputCol("filteredWords").setStopWords(noHello)
    val noStopWordsDFCustom = removerCustom.transform(wordsDF)
    noStopWordsDFCustom.select("sentence", "filteredWords").show(10, false)

    /**
      * 3.NGrams
      * NGrams are word combinations created as sequences of words.
      * N stands for the number of words in the sequence. setN() is used to specify the value of N
      */

    val ngram = new NGram().setInputCol("filteredWords").setOutputCol("ngrams").setN(2)
    val nGramDF = ngram.transform(noStopWordsDF)
    nGramDF.show(10, false)

    /**
      * 4.TF-IDF
      * TF-IDF stands for term frequency-inverse document frequency,
      * which measures how important a word is to a document in a collection of documents
      */

    //HashingTF is a transformer, which takes a set of terms and converts them into vectors of fixed length
    //by hashing each term using a hash functionto generate an index for each term
    val hashingTF = new HashingTF().setInputCol("filteredWords").setOutputCol("rawFeatures").setNumFeatures(100)
    val rawFeaturesDF = hashingTF.transform(noStopWordsDF)

    //IDF is an estimator, which is fit onto a dataset and then generates features by scaling the input features.
    //IDF works on output of a HashingTF transformer

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(rawFeaturesDF)
    val featuresDF = idfModel.transform(rawFeaturesDF)
    featuresDF.select("id", "filteredWords", "rawFeatures", "features").show(10, false)

    /**
      * 5.Word2Vec
      */

    val word2Vec = new Word2Vec().setInputCol("words").setOutputCol("wordvector").setVectorSize(3).setMinCount(0)
    val word2VecModel = word2Vec.fit(noStopWordsDF)
    val word2VecDF = word2VecModel.transform(noStopWordsDF)

    /**
      * 6.CountVectorizer
      * is used to convert a collection of text documents to vectors of token counts
      * producing sparse representations for the documents over the vocabulary
      * the end result is a vector of features, which can then be passed to other algorithms
      */
    val countVectorizer = new CountVectorizer().setInputCol("filteredWords").setOutputCol("features")
    val countVectorizerModel = countVectorizer.fit(noStopWordsDF)
    val countVectorizerDF = countVectorizerModel.transform(noStopWordsDF)
}
