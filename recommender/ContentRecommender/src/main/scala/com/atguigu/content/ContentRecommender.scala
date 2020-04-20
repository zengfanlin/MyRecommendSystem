package com.atguigu.content

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.SparkSession
import org.jblas.DoubleMatrix

case class Product(productId: Int, name: String, imageUrl: String, categories: String, tags: String)

case class MongoConfig(uri: String, db: String)

// 定义标准推荐对象
case class Recommendation(productId: Int, score: Double)

// 定义商品相似度列表
case class ProductRecs(productId: Int, recs: Seq[Recommendation])

object ContentRecommender {
  // 定义mongodb中存储的表名
  val MONGODB_PRODUCT_COLLECTION = "Product"
  val CONTENT_PRODUCT_RECS = "ContentBasedProductRecs"

  def main(args: Array[String]): Unit = {
    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://cdh03:27017/recommender",
      "mongo.db" -> "recommender"
    )
    // 创建一个spark config
    val sparkConf = new SparkConf().setMaster(config("spark.cores")).setAppName("ContentRecommender")
    sparkConf.set("spark.executor.heartbeatInterval", "1000")
    // 创建spark session
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    import spark.implicits._
    implicit val mongoConfig = MongoConfig(config("mongo.uri"), config("mongo.db"))
    // 载入数据，做预处理
    val productTagsDF = spark.read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_PRODUCT_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[Product]
      .map(
        x => (x.productId, x.name, x.tags.map(c => if (c == '|') ' ' else c))
      )
      .toDF("productId", "name", "tags")
      .cache()
    // TODO: 用TF-IDF提取商品特征向量
    // 1. 实例化一个分词器，用来做分词，默认按照空格分
    val tokenizer = new Tokenizer().setInputCol("tags").setOutputCol("words")

    // 用分词器做转换，得到增加一个新列words的DF
//    |productId|                name|                tags|               words|
//    +---------+--------------------+--------------------+--------------------+
//    |   259637|                小狗钱钱|书 少儿图书 教育类 童书 不错 ...|[书, 少儿图书, 教育类, 童书...|
//    |     3982|Fuhlen 富勒 M8眩光舞者时...|  富勒 鼠标 电子产品 好用 外观漂亮|[富勒, 鼠标, 电子产品, 好用...|
    val wordsDataDF = tokenizer.transform(productTagsDF)
//    wordsDataDF.show()

    // 2. 定义一个HashingTF工具，计算频次
//    |productId|                name|                tags|               words|         rawFeatures|
//    +---------+--------------------+--------------------+--------------------+--------------------+
//    |   259637|                小狗钱钱|书 少儿图书 教育类 童书 不错 ...|[书, 少儿图书, 教育类, 童书...|(800,[67,259,267,...|
//    |     3982|Fuhlen 富勒 M8眩光舞者时...|  富勒 鼠标 电子产品 好用 外观漂亮|[富勒, 鼠标, 电子产品, 好用...|(800,[350,502,510...|
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(800)//特征数量为800
    val featurizedDataDF = hashingTF.transform(wordsDataDF)
//    featurizedDataDF.show()
    // 3. 定义一个IDF工具，计算TF-IDF
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
//    // 训练一个idf模型
    val idfModel = idf.fit(featurizedDataDF)
    // 得到增加新列features的DF
    val rescaledDataDF = idfModel.transform(featurizedDataDF)
//    rescaledDataDF.show(truncate = false)
    // 对数据进行转换，得到RDD形式的features
    val productFeatures = rescaledDataDF.map{
      row => ( row.getAs[Int]("productId"), row.getAs[SparseVector]("features").toArray )
    }
      .rdd
      .map{
        case (productId, features) => ( productId, new DoubleMatrix(features) )
      }
    // 两两配对商品，计算余弦相似度
        val productRecs = productFeatures.cartesian(productFeatures)
          .filter{
            case (a, b) => a._1 != b._1
          }
          // 计算余弦相似度
          .map{
            case (a, b) =>
              val simScore = consinSim( a._2, b._2 )
              ( a._1, ( b._1, simScore ) )
          }
          .filter(_._2._2 > 0.4)
          .groupByKey()
          .map{
            case (productId, recs) =>
              ProductRecs( productId, recs.toList.sortWith(_._2>_._2).map(x=>Recommendation(x._1,x._2)) )
          }
          .toDF()
        productRecs.write
          .option("uri", mongoConfig.uri)
          .option("collection", CONTENT_PRODUCT_RECS)
          .mode("overwrite")
          .format("com.mongodb.spark.sql")
          .save()

    spark.stop()
  }

  def consinSim(product1: DoubleMatrix, product2: DoubleMatrix): Double = {
    product1.dot(product2) / (product1.norm2() * product2.norm2())
  }
}
