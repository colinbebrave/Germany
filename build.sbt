name := "Germany"

version := "0.1"

scalaVersion := "2.11.8"

val sparkVersion = "2.3.1"

libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core" % sparkVersion,
    "org.apache.spark" %% "spark-streaming" % sparkVersion,
    "org.apache.spark" %% "spark-sql" % sparkVersion,
    "org.apache.spark" %% "spark-hive" % sparkVersion,
    "org.apache.spark" %% "spark-mllib" % sparkVersion,
    "edu.stanford.nlp" % "stanford-corenlp" % "3.9.1",
    "org.ansj" % "ansj_seg" % "5.1.6"
)
