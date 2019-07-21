import warnings
warnings.filterwarnings("ignore", category=Warning)

from pyspark.sql import SparkSession
from pyspark.sql.functions import round
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoderEstimator
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline


# start a sparksession
spark = SparkSession.builder \
                    .master('local[*]') \
                    .appName('test') \
                    .getOrCreate()


# read the dataset in pyspark
df = spark.read.csv('flights-data.csv', inferSchema=True, sep=',', header=True, nullValue='NA')

# remove the flight columns
df = df.drop('flight')

# there are some null values in this dataset, Remove it
df = df.dropna()

'''
Lets considers a flight to be "delayed" if it arrives late after 15 minutes than it was sheduled. 
Also, the mile column has data into miles, so lets change this to KM.
'''
df_withKM = df.withColumn('KM', round(df.mile * 1.60934, 0)).drop('mile')

# create a extra column called label based on whether there has been delay or not and assigned to 0 and 1
flight_data = df_withKM.withColumn('label', (df_withKM.delay >= 15).cast('integer'))

flight_data = flight_data.drop('delay')

# Change those catagorical text columns which is going to be used in ML to catagorical numerical columns
indexers = StringIndexer(inputCol='carrier', outputCol='carrier_idx')

# do one hot coding for catagorical numerical columns
onehot = OneHotEncoderEstimator(
    inputCols=['dow', 'carrier_idx'],
    outputCols=['dow_d', 'carrier_d']
)

# Extract the features columns
assembler = VectorAssembler(inputCols=['mon', 'dom', 'dow_d', 'carrier_d', 'KM', 'duration'],
                            outputCol='features')


# split data into train, test using 80% train and 20% test. Also assigns the random speed
x_train, x_test = flight_data.randomSplit([0.80, 0.20], seed=42)


# create a GBT classifier
gbt = GBTClassifier(labelCol= 'label')

pipeline = Pipeline(stages=[indexers,  onehot, assembler, gbt])

model = pipeline.fit(x_train)

prediction = model.transform(x_test)


# Create a confusion matrix
prediction.groupBy('label', 'prediction').count().show()

# Calculate the elements of the confusion matrix
TN = prediction.filter('prediction = 0 AND label = prediction').count()
TP = prediction.filter('prediction = 1 AND label = prediction').count()
FN = prediction.filter('prediction = 0 AND label != prediction').count()
FP = prediction.filter('prediction = 1 AND label != prediction').count()

# Accuracy measures the proportion of correct predictions
accuracy = (TN + TP) / (TN + TP + FN + FP)
print(accuracy)
'''
see the accuracy and also the confusion matrix. Confusion matrix can be very helpful on finding how the model is doing
and how well it predicts false positive and false negative. Accuracy can be bias if there imbalanced dataset, 
Confusion  matrix is very helpful on such situations. 
'''
# do cross validation for this model

params = (ParamGridBuilder()
             .addGrid(gbt.maxDepth, [2, 4])
             .addGrid(gbt.maxBins, [10, 20])
             .addGrid(gbt.maxIter, [5, 10])
             .build())

evaluator = BinaryClassificationEvaluator(labelCol='label')

cv = CrossValidator(estimator=pipeline, estimatorParamMaps= params, evaluator= evaluator, numFolds=5)

cvmodel = cv.fit(x_train)

best_model = cvmodel.bestModel

print(evaluator.evaluate(best_model.transform(x_test)))

'''
We can also get other number of features like best params using best_model.stages[3].extractParamMap(). Here [3] means
the stages which is at index 3 , i.e our classification algorithms. 
'''