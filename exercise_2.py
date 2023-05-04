from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import stddev
from pyspark.sql.functions import col
from pyspark.sql.functions import explode


# Create a SparkSession
spark = SparkSession.builder.appName('movie-recommender').getOrCreate()

# Load the ratings data
ratings_df = spark.read \
    .format("csv") \
    .option("header", "false") \
    .option("delimiter", "\t") \
    .load("ml-100k/u.data") \
    .toDF("user_id", "movie_id", "rating", "timestamp")

# Convert the user_id column to integer
ratings_df = ratings_df \
    .withColumn("user_id", ratings_df["user_id"].cast("integer")) \
    .withColumn("movie_id", ratings_df["movie_id"].cast("integer")) \
    .withColumn("rating", col("rating").cast("double"))

# Split the dataset into training and test sets
(training_data, test_data) = ratings_df.randomSplit([0.9, 0.1])

# Build the recommendation model using ALS
als = ALS(maxIter=20, regParam=0.01, userCol="user_id", itemCol="movie_id", ratingCol="rating",
          coldStartStrategy="drop")

model = als.fit(training_data)

predictions = model.transform(test_data)
evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
evaluator_mae = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")
rmse = evaluator_rmse.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)
stddev_ = predictions.select(stddev('rating')).collect()[0][0]

print("\n\nRoot-mean-square error = " + str(rmse))
print("Mean absolute error = " + str(mae))
print("Standard deviation of ratings = " + str(stddev_))

#############################################################

userRecs = model.recommendForAllUsers(10)

userRecs = userRecs.select('user_id', explode('recommendations').alias('rec'))
userRecs = userRecs.select('user_id', 'rec.movie_id', 'rec.rating')

userRecs.show()

