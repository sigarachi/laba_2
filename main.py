import nltk
from nltk.stem import PorterStemmer
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, explode, col, lower, regexp_extract, udf
from pyspark.sql.types import StringType
import findspark

findspark.init()

spark = (SparkSession.builder
         .appName("The top most common words in The Master and Margarita, by Bulgakov Mikhail")
         .master("local[*]")
         .getOrCreate()
         )

book = spark.read.text("Bulgakov Mikhail. The Master and Margarita.txt")

with open("stop_words_english.txt", "r") as f:
    text = f.read()
    stopwords = text.splitlines()

print(len(stopwords), stopwords[:15])

lines = book.select(split(book.value, " ").alias("line"))
lines.show(5)

words = lines.select(explode(col("line")).alias("word"))
words.show(15)

words_lower = words.select(lower(col("word")).alias("word_lower"))
words_lower.show()

words_clean = words_lower.select(
    regexp_extract(col("word_lower"), "[a-z]+", 0).alias("word")
)

words_clean.show()

words_nonull = words_clean.filter(col("word") != "")
words_nonull.show()

words_without_stopwords = words_nonull.filter(
    ~words_nonull.word.isin(stopwords))

words_count_before_removing = words_nonull.count()
words_count_after_removing = words_without_stopwords.count()

print(words_count_before_removing, words_count_after_removing)

top_words_count = (words_without_stopwords.groupby("word")
                   .count()
                   .orderBy("count", ascending=False)
                   )

rank = 50
top_words_count.show(rank)

least_words_count = (words_without_stopwords.groupby("word")
                     .count()
                     .orderBy("count", ascending=True)
                     )

least_words_count.show(rank)


def stem(str_in):
    ps = PorterStemmer()
    res_str = ps.stem(str_in)
    return res_str


stemUDF = udf(lambda z: stem(z), StringType())


stemmed_top = top_words_count.withColumn("word", stemUDF(top_words_count["word"]))

stemmed_top.show(rank)

stemmed_low = least_words_count.withColumn("word", stemUDF(least_words_count["word"]))

stemmed_low.show(rank)
