class ModelTransformer():
	
	def __init__(self, model):
		self.model = model
		
	def fit(self, *args, **kwargs):
		self.model.fit(*args, **kwargs)
		return self
	
	def transform(self, X, **transform_parameters):
		return DataFrame(self.model.predict(X))

		
		
def returnTransformers():
	transformers = [
		None, 
		MaxAbsScaler(), 
		MinMaxScaler(), 
		RobustScaler(), 
		StandardScaler(),  
		Normalizer(norm = 'l1'), 
		Normalizer(norm = 'l2'), 
		Normalizer(norm = 'max')
	]


def returnModels():
	models = {
		"LogisticRegression" 		:	[None, "n_jobs=-1", "random_state=1"],
		"LinearRegression"			:	[None, None],
		"RandomForestClassifier"	:	[None, "n_estimators=25", "random_state=1"],
		"GaussianNB"				:	[None, None],
		"CalibratedClassifierCV"	: 	["clf,method='sigmoid',cv='prefit'", "clf,method='isotonic',cv='prefit'"],
		"VotingClassifier"			:	[None, None, None]
	}
def exampleWorkFlow():

	pipeline = Pipeline([
		('features', FeatureUnion([
			('continuous', Pipeline ([
				('extract', ColumnExtractor(CONTINUOUS_FIELDS)),
				('scale', Normalizer())
			])),
			('factors', Pipeline([
				('extract', ColumnExtractor(FACTOR_FIELDS)),
				('one_hot', OneHotEncoder(n_values=5)),
				('to_dense'), DenseTransformer())
			])),
			('weekday', Pipeline ([
				('extract', ColumnExtractor(FACTOR_FIELDS)),
				('one_hot', OneHotEncoder()),
				('to_dense'), DenseTransformer())
			])),
			('hour_of_day', HourOfDayTransformer()),
			('month', Pipeline([
				('extract', ColumnExtractor(FACTOR_FIELDS)),
				('to_month', DataTransformer()),
				('one_hot', OneHotEncoder()),
				('to_dense'), DenseTransformer())
			])),
			('growth', Pipeline([
				('datetime', ColumnExtractor(['datetime'])),
				('to_numeric', MatrixConversion(int)),
				('regression', ModelTransformer(LinearRegression()))
			]))
		])),
		('estimators', FeatureUnion([
			('knn', ModelTransformer(KNeighborsRegressor(n_neighbors=5))),
			('gbr', ModelTransformer(GradientBoostingRegressor())),
			('dtr', ModelTransformer(DecisionTreeRegressor())),
			('etr', ModelTransformer(ExtraTreesRegressor())),
			('rfr', ModelTransformer(RandomForestRegressor())),
			('par', ModelTransformer(PassiveAggressiveRegressorRegressor())),
			('en', ModelTransformer(ElasticNet())),
			('cluster', ModelTransformer(KMeans(n_clusters=2))),
		])),
		('estimator', KNeighborsRegressor())
	])


	

def presets():
	# these can be custom sets of models and parameters that have, historically, worked well on a given dataset, eg: numerai logloss
	NUMERAI = {
	
	}
	
	KAGGLE = {
	
	}
	
	DATADRIVEN = {
	
	}
	
	TEXT_CLASSIFICATION = Pipeline([
		('extract_words', extractWords()),
		('features', FeatureUnion([
			('ngram_tf_idf', Pipeline([
				('counts', CountVectorizer()),
				('tf_idf', TfidfTransformer()),
			])),
			('doc_length', LengthTransformer()),
			('misspellings', MispellingCountTransformer())
		])),
		('classifier', MultinomialNB())	
	])
	
def requestPipeline(pipeline):

	# define featureUnions and pipelines here
	pipes = {
		# numerai : Pipeline(('pca',pca),('lr',LogisticRegression)),
		# regression: Pipeline(('pca',pca),('lr',LogisticRegression)),
		# classification : Pipeline(('pca',pca),('lr',LogisticRegression))
	}
	
	if pipeline in pipes.keys():
		p = pipes[pipeline]
		return p
	return 0
	















