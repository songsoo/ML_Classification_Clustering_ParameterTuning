# ML_Classification_Clustering_ParameterTuning
<H1>How to use</H1> <br>
<H3>Settings before clustering & classification</H3>
a. import ML_TP_auto.py<br>
b. Read dataset file <br>
c. Divide dataset into Target and Features <br>
d. choose columns to use for clustering and classification <br><br>
<H3>Auto ML</H3>
<H3>1. classification</H3>
ML_TP_auto.get_Result(x_train,y_train,x_test,y_test,numerical_columns,categorical_columns)<br>
x_train: feature to train classification models (pd.Dataframe) <br>
y_train: target to train classification models (pd.Dataframe)<br>
x_test: feature to test classification models (pd.Dataframe)<br>
y_test: target to test classification models (pd.Dataframe)<br>
numerical_columns : numerical column's names sets (2d-array)<br>
categorical_columns : categorical column's names sets (2d-array)<br><br>
return : test target and test features's score from the higest score of each models with train target and train features and each model's parameters tuned

&nbsp;<H3>2. clustering</H3>
&nbsp;ML_TP_auto.findBest(or_data, y, numerical_columns, categorical_columns, max_cluster=None, n_inits=None, max_iters=None,tols=None, verboses=None, covariance_types=None,numlocals=None, max_neighbors=None, epsS=None, min_samples=None, metrics=None, algorithms=None, leaf_sizes=None, bandwidths=None, n_job=None)<br>
<H4>parameters </H4>           
or_data : feature (pd.Dataframe)<br>
y : target (pd.Dataframe)<br>
numerical_columns : numerical column's names sets (2d-array)<br>
categorical_columns : categorical column's names sets (2d-array)<br>
max_cluster : default set for parameter, can be None (integer) default=6<br>
n_inits : default set for parameter, can be None (integer) default=[5,10,15,20]<br>
max_iters :  default set for parameter, can be None (integer) default=300 <br>
tols :  default set for parameter, can be None (float) default=1e-4<br>
verboses :  default set for parameter, can be None (integer) default=0 <br>
covariance_types:  default set for parameter, can be None (String) ex) {‘full’, ‘tied’, ‘diag’, ‘spherical’}, default=’full’<br>
numlocals : default set for parameter, can be None (integer)<br>
max_neightbors : default set for parameter, can be None (integer)<br>
epsS : default set for parameter, can be None (float) default=[0.1,0.5,0.7,0]<br>
min_samples : default set for parameter, can be None (integer) default=[3,4,5]<br>
metrics : default set for parameter, can be None (String) default=['euclidean']<br>
algorithms : default set for parameter, can be None (String) default=['auto']<br>
bandwidths : default set for parameter, can be None (integer) default=[estimate_bandwidth(X, quantile=0.25), estimate_bandwidth(X, quantile=0.50), estimate_bandwidth(X, quantile=0.75)]<br>
n_job: default set for parameter, can be None (integer) default=-1<br><br>
return : none <br>
it only shows the result <br>

you can find example for using it in 'ML_TP_main.py'

<br><br>

