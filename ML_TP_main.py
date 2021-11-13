import pandas as pd
import ML_TP_auto
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns',None)

#Read CSV file
data = pd.read_csv("online_shoppers_intention.csv")

#Divide dataset into Target(y) and Features(x)
y = data.loc[:,'Revenue']
x = data.loc[:,['Administrative','Administrative_Duration','Informational','Informational_Duration','ProductRelated','ProductRelated_Duration','BounceRates','ExitRates','ExitRates','PageValues','SpecialDay','Month','OperatingSystems','Browser','Region','TrafficType','VisitorType','Weekend','Revenue']]

#-------------------------------------------------------------------------------------------------------------
#columns to use for clustering
c_numerical_columns=[['Administrative_Duration', 'ProductRelated_Duration'], ['ProductRelated', 'ProductRelated_Duration'], ['Administrative_Duration', 'ProductRelated', 'ProductRelated_Duration']]
c_categorical_columns=[[],[],[]]

# # Clustering
ML_TP_auto.findBest(x,y,c_numerical_columns,c_categorical_columns)

#-------------------------------------------------------------------------------------------------------------
#columns to use for classification
numerical_columns = [['Administrative','Administrative_Duration'],['Administrative','Administrative_Duration','ProductRelated','ProductRelated_Duration'],['Administrative','Administrative_Duration','ProductRelated','ProductRelated_Duration','PageValues']]
categorical_columns = [['Month','OperatingSystems'],['Month','OperatingSystems','Weekend'],['Month','OperatingSystems','Weekend']]

# Classification
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 100, shuffle=True)
result = ML_TP_auto.get_Result(x_train,y_train,x_test,y_test,numerical_columns,categorical_columns)
print("Score :" ,result)




