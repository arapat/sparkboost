name := "SparkBoost"

version := "0.1"

scalaVersion := "2.11.7"

scalacOptions += "-optimize"

libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core" % "2.0.2",
    "org.apache.spark" %% "spark-sql" % "2.0.2",
    "org.apache.spark" %% "spark-mllib" % "2.0.2"
)
