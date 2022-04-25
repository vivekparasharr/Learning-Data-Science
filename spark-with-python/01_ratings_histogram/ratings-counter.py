
# SparkConf - required to create sparkcontext as it allows us to configure spark context
# SparkContext - fundamental starting point that spark gives you to create rdd's from
from pyspark import SparkConf, SparkContext
import collections

# tell spark that we will be running on a local node just on this system
# there are extensions to local that tell spark to split up the processing between cpu cores, but we are not doing that here
# we also set the app name so we can track the job
conf = SparkConf().setMaster("local").setAppName("RatingsHistogram")
sc = SparkContext(conf = conf)

# load the data file into an rdd
lines = sc.textFile("file:///Users/vivekparashar/Documents/Code/Big-Data/Spark with Python/01_ratings_histogram/ml-100k/u.data")
# split into user id, movie id, rating and timestamp, then pick the rating (field number 2) -> saves it to new rdd called ratings
ratings = lines.map(lambda x: x.split()[2])
# perfrom the action - countbyValue
result = ratings.countByValue()

sortedResults = collections.OrderedDict(sorted(result.items()))
for key, value in sortedResults.items():
    print("%s %i" % (key, value))
 