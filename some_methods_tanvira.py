# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 12:09:00 2023

@author: RimJhim
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from datetime import date
import seaborn as sb
from scipy import stats
#from scipy import norm
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import train_test_split
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.preprocessing import StandardScaler
##pip install category_encoders
#from category_encoders.one_hot import OneHotEncoder

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error

from numpy import mean
from numpy import std

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Lasso



df_raw = pd.read_csv("./train.csv")

df_raw.describe()
df_raw.shape  #6495 rows 55 columns
df_raw.dtypes

##checking for missing values
df_raw.isna().sum()
##property square feet has a lot of missing values-6333 of 6495 rows is empty
##host response time and response rate is 1461
##for reviews 1290 to 1405 rows are missing
pct_miss= (df_raw.isnull().sum() / len(df_raw))*100

##use log of price for target
df_raw['lprice']=np.log(df_raw['target'])

##Descriptive statistics of some categorical variables
print(df_raw['property_type'].unique())
print(df_raw['property_room_type'].unique())
print(df_raw['property_bed_type'].unique())

df_raw['property_type'].value_counts()
df_raw['property_room_type'].value_counts()
df_raw['property_bed_type'].value_counts()

df_raw['host_since2'] = pd.to_datetime(df_raw['host_since'])
##only 1 row of host_since is missing; drop it to compute number of days/yrs
##sb has been host
df_raw['date_today']=pd.to_datetime(date.today())
df2 = df_raw.dropna(subset=['host_since'])

df2['host_y']=((df2['date_today']-df2['host_since2'])).dt.days
df2.columns


##seeing which types of property have highest mean target and counts 
df2.groupby('property_type')['target'].mean()
df2.groupby(['property_type'])['target'].median()
df2.groupby('property_type')['target'].count()

df2.groupby('property_room_type')['target'].mean()
df2.groupby(['property_room_type'])['target'].median()
df2.groupby('property_room_type')['target'].count()



df2.groupby('booking_cancel_policy')['target'].mean()
df2.groupby('booking_cancel_policy')['target'].median()
df2.groupby('booking_cancel_policy')['target'].count()

##getting some correlations
df3 = df2[['property_max_guests','property_bathrooms','property_bedrooms','property_beds',
   'booking_price_covers','booking_min_nights',            
  'booking_max_nights','booking_availability_30',        
 'booking_availability_60','booking_availability_90','booking_availability_365','lprice']]

corr = df3.corr()
sb.heatmap(corr, cmap="Blues", annot=True)

##dropping missing rows and getting correlations 
df4=df3.dropna()
corr2 = df4.corr()
sb.heatmap(corr2, annot=True,linewidth=2)

##checking if log of price is normal or not
sb.distplot(df_raw['lprice'], fit=norm)
plt.show()
stats.probplot(df_raw['lprice'], dist="norm",plot=plt)
plt.show()
##some divergence at the tails, also some large values(histogram)-
##can normal distrubution assumption still be made? 

#Host response time has missing values and 
##it is categorical-how to impute it
##need to encode with nan values and impute
##knnimputer can be used 

##LabelEncoding
#dfl=pd.DataFrame(df_raw['host_response_time'])
#m = dfl.isnull()
#dfl2= dfl.astype(str).apply(LabelEncoder().fit_transform)
#dfl3=dfl2.where(~m,dfl)
##Encoding with Nan on host_response_time remaining
##One hot encoding for categorical variables
##One hot encoder does not work with nan values-encoded to new values 


###
X=df2.copy()
X.drop(['property_id','property_name',
               'property_summary','property_space',
                  'property_desc','property_neighborhood',
                 'property_notes','property_transit',
                'property_access','property_interaction',
                 'property_rules','property_zipcode',
                   'property_lat','property_lon',
                'property_amenities',
                'property_scraped_at','property_last_updated',
            'host_id','host_since',
        'host_location','host_about',
                 'host_response_time','host_verified',
                 'reviews_first','date_today',
                 'reviews_last',
                'extra',
             'target','host_since2','lprice',
         ],axis=1,inplace=True)
X.columns
y=df2[['lprice']]

X['property_type'].value_counts()
X['property_room_type'].value_counts()
X['property_bed_type'].value_counts()
X['booking_cancel_policy'].value_counts()

##Do new categorization of property type as some categories only have 1 observation
##Also get uneven categories after train test split
##All obs with values 7 or below are recoded as Other
##The next largest number of observations above 7 is 19
ptype= {'Apartment':'Apartment'
                    ,'House':'House'
                    ,'Bed & Breakfast':'Bed & Breakfast'
                    ,'Townhouse':'Townhouse'
                    ,'Condominium':'Condomium'
                    ,'Guesthouse':'Guesthouse'
                    ,'Other':'Other'
                    ,'Villa':'Villa'
                    ,'Cabin':'Other'
                    ,'Serviced apartment':'Other'
                    ,'Castle':'Other'                  
                    ,'Dorm':'Other'                    
                    ,'Guest suite':'Other'
                    ,'Hostel':'Other'                   
                    ,'Boutique hotel':'Other'          
                    ,'Earth House':'Other'              
                    ,'Camper/RV':'Other'               
                    ,'Yurt':'Other'                     
                    ,'Chalet':'Other'                   
                    ,'Boat':'Other'                     
                    ,'Tent':'Other'                     
                    ,'Timeshare':'Other'}
X=X.assign(p = X.property_type.map(ptype))

X_train, X_validation,y_train,y_validation= train_test_split(X,y, test_size=0.20, 
        random_state=0)







 
cat_var_train=X_train[['property_bed_type','property_room_type',
            'p','booking_cancel_policy']]

cat_var_valid=X_validation[['property_bed_type','property_room_type',
            'p','booking_cancel_policy']]

cat_train = cat_var_train.merge(pd.get_dummies(cat_var_train.p, drop_first=True), left_index=True, right_index=True)
cat_train.drop('p', axis = 1, inplace=True)

cat_train2=cat_train.merge(pd.get_dummies(cat_train.property_bed_type, drop_first=True), left_index=True, right_index=True)
cat_train2.drop('property_bed_type', axis = 1, inplace=True)

cat_train3=cat_train2.merge(pd.get_dummies(cat_train2.property_room_type, drop_first=True), left_index=True, right_index=True)
cat_train3.drop('property_room_type', axis = 1, inplace=True)

cat_train4=cat_train3.merge(pd.get_dummies(cat_train3.booking_cancel_policy, drop_first=True), left_index=True, right_index=True)
cat_train4.drop('booking_cancel_policy', axis = 1, inplace=True)

#cat_train4=cat_train4.drop(cat_train4.columns[[1,4,5,8,16,19]], axis=1)

#encoded=pd.DataFrame(encoder.fit_transform(cat_var_train))
#encoded.columns.value_counts()

#encoded2=pd.DataFrame(encoder.fit_transform(cat_var_valid))
##problem with validation set-validation set has an extra booking
##cancel policy level that the training set does not have-
##get get more columns in validation set- this cancellation policy level
##makes up only 1 row in dataset and so drop it
#encoded2=encoded2.iloc[:,0:24]

cat_valid = cat_var_valid.merge(pd.get_dummies(cat_var_valid.p, drop_first=True), left_index=True, right_index=True)
cat_valid.drop('p', axis = 1, inplace=True)

cat_valid2=cat_valid.merge(pd.get_dummies(cat_valid.property_bed_type, drop_first=True), left_index=True, right_index=True)
cat_valid2.drop('property_bed_type', axis = 1, inplace=True)

cat_valid3=cat_valid2.merge(pd.get_dummies(cat_valid2.property_room_type, drop_first=True), left_index=True, right_index=True)
cat_valid3.drop('property_room_type', axis = 1, inplace=True)

cat_valid4=cat_valid3.merge(pd.get_dummies(cat_valid3.booking_cancel_policy, drop_first=True), left_index=True, right_index=True)
cat_valid4.drop('booking_cancel_policy', axis = 1, inplace=True)


cat_valid4.drop('super_strict_30', axis = 1, inplace=True) ##only 1 row in entire data
cat_train4.columns
cat_valid4.columns

#cat_train4=cat_train4.drop(cat_train4.columns[[1,4,5,8,16,28]], axis=1)

 

dfnew=X_train.copy()
dfnewv=X_validation.copy()
##dropping encoded variables and missing numeric col rows from original train and validation data 
dfnew.drop(['property_room_type','property_bed_type','property_type','p',
              'booking_cancel_policy','property_sqfeet','host_response_rate',
            'property_bathrooms', 'property_bedrooms', 'property_beds',
            'reviews_rating', 'reviews_acc', 'reviews_cleanliness',
            'reviews_checkin', 'reviews_communication', 'reviews_location',
            'reviews_value', 'reviews_per_month',], 
           axis=1, inplace=True)

dfnewv.drop(['property_room_type','property_bed_type','property_type','p',
              'booking_cancel_policy','property_sqfeet','host_response_rate',
            'property_bathrooms', 'property_bedrooms', 'property_beds',
            'reviews_rating', 'reviews_acc', 'reviews_cleanliness',
            'reviews_checkin', 'reviews_communication', 'reviews_location',
            'reviews_value', 'reviews_per_month',], 
           axis=1, inplace=True)

##dropping additional unnecessary columns


dfnew.columns  ##dataframe with nonmissing numeric values
dfnewv.columns


##standardize the variables in dfnew
scaler = StandardScaler()

scaled_df = scaler.fit_transform(dfnew)
scaled_df = pd.DataFrame(scaled_df,columns = dfnew.columns)  ##scaled numeric columns with nonmissing values

scaled_df_valid=scaler.fit_transform(dfnewv)
scaled_df_valid = pd.DataFrame(scaled_df_valid,columns=dfnewv.columns) 



##1. note:need to drop missing rows and do regressions again





##2.try imputing 

###host response time impute using LabelEncoder and KNN Imputer-to do
##host response time not included in these regressions 
#imputer = KNNImputer(n_neighbors=1)
#hrt=imputer.fit_transform(dfl3)




##use knn imputer-use 5 and 10 neighbors for numeric cols with missing rows
df_small_t=X_train.filter(['property_bathrooms','property_bedrooms',
                'property_beds','host_response_rate',
                'reviews_rating','reviews_acc','reviews_cleanliness',
                    'reviews_checkin','reviews_communication',
                    'reviews_location','reviews_value','reviews_per_month'])

df_small_v=X_validation.filter(['property_bathrooms','property_bedrooms',
                'property_beds','host_response_rate',
                'reviews_rating','reviews_acc','reviews_cleanliness',
                    'reviews_checkin','reviews_communication',
                    'reviews_location','reviews_value','reviews_per_month'])

df_small_t.isna().sum()
df_small_v.isna().sum()
##missing values in both training and validation sets 


##5 neighbors
scaler2 = MinMaxScaler(feature_range=(0, 1))
sfit = pd.DataFrame(scaler2.fit_transform(df_small_t), columns = df_small_t.columns)

imputer5 = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
df_knn_imputed = pd.DataFrame(imputer5.fit_transform(sfit), columns=df_small_t.columns)


sfitv = pd.DataFrame(scaler2.fit_transform(df_small_v), columns = df_small_v.columns)


df_knn_imputed_v = pd.DataFrame(imputer5.fit_transform(sfitv), columns=df_small_v.columns)


##10 neighbors


imputer10 = KNNImputer(n_neighbors=10, weights='uniform', metric='nan_euclidean')
df_knn_imputed10 = pd.DataFrame(imputer10.fit_transform(sfit), columns=df_small_t.columns)



df_knn_imputed_v10 = pd.DataFrame(imputer10.fit_transform(sfitv), columns=df_small_v.columns)

##upto now I have not taken into account if there 
##are any outliers or infleuntial observations

##Merging imputed datafrane with dataframe with ecoded columns
dfm=pd.concat([scaled_df,df_knn_imputed],axis=1, join='inner')
#dfm=pd.merge(scaled_df,df_knn_imputed,left_index=True,right_index=True)
dfm2 = pd.concat([dfm.reset_index(drop = True),cat_train4.reset_index(drop=True)],
                           axis = 1)

##validation set merge

dfmv=pd.concat([scaled_df_valid,df_knn_imputed_v],axis=1, join='inner')
#dfm=pd.merge(scaled_df,df_knn_imputed,left_index=True,right_index=True)
dfm2v = pd.concat([dfmv.reset_index(drop = True),cat_valid4.reset_index(drop=True)],
                           axis = 1)

###doing linear regression

##linear regression-no grid search-all features
lr = LinearRegression()
lr.fit(dfm2, y_train)
y_hat_train = lr.predict(dfm2)
y_hat_validation=lr.predict(dfm2v)


print(f"r^2: {r2_score(y_train, y_hat_train)}")
print(f"rmse: {np.sqrt(mean_squared_error(y_train, y_hat_train))}")
print(f"mae: {mean_absolute_error(y_train, y_hat_train)}")

##
#r^2: 0.007105506326830824
#rmse: 0.5574609858065449
#mae: 0.4330964210250904

##error on validation set
print(f"rmse: {np.sqrt(mean_squared_error(y_validation, y_hat_validation))}")
print(f"mae: {mean_absolute_error(y_validation, y_hat_validation)}")

##Rmse and mae on validation set
#rmse: 0.5701358673627932
#mae: 0.44027454237928904


##linear regression with grid search 


##feature selection with lasso
la= Lasso()
alphas=dict()
alphas['alpha']=[1.1,1,0.1,0.01,0.001,0.0001,0.001,0]

grid_search_lasso = GridSearchCV(estimator=la,param_grid=alphas,scoring='neg_mean_squared_error',
                cv=5,n_jobs=-1)

grid_search_lasso.fit(dfm2,y_train)

print('Best Score: %s' % grid_search_lasso.best_score_)
print('Best Hyperparameters: %s' % grid_search_lasso.best_params_)

##coefficients and importance
la2 = Lasso(alpha=0.01)
#
# Fit the Lasso model
#
la2.fit(dfm2, y_train)
#
# Create the model score

la2.score(dfm2, y_train)
la2.score(dfm2v, y_validation)

la2.coef_

df_small_la2=dfm2.iloc[:,[5,-1]] ##chossing these cols from coefficient values 
df_small_la2v=dfm2v.iloc[:,[5,-1]]

la3 = Lasso(alpha=0.01)

##fit lasso model with selected features 

la2.fit(df_small_la2, y_train)
pred_l_tr=la2.predict(df_small_la2)
pred_l_v=la2.predict(df_small_la2v)

print(f"rmse_train_dr: {np.sqrt(mean_squared_error(y_train,pred_l_tr))}")
print(f"mae_train_dr: {mean_absolute_error(y_train,pred_l_tr)}")

##RMSE and MAE train 
#rmse_train: 0.559185350111474
#mae_train: 0.4344480310202465
#
print(f"rmse_validation_l: {np.sqrt(mean_squared_error(y_validation,pred_l_v))}")
print(f"mae_validation_l: {mean_absolute_error(y_validation,pred_l_v)}")

##Rmse and Mae
#rmse_validation_l: 0.5674127799463126
#mae_validation_l: 0.4396895039504218




##Recursive feature selection to select best features for model
##RFECV automatically selects number of features that will give best result

##Using DecisionTreeRegressor as model and using Recursive feature selection
##to get best features 
##evaluating by gridsearchCV

rfs = RFECV(estimator=DecisionTreeRegressor())
model = DecisionTreeRegressor()
pipeline = Pipeline(steps=[('s',rfs),('m',model)])
pipeline.fit(dfm2, y_train)
predict_y_dr= pipeline.predict(dfm2)
print(f"rmse_train_dr: {np.sqrt(mean_squared_error(y_train,predict_y_dr))}")
print(f"mae: {mean_absolute_error(y_train, predict_y_dr)}")

##RMSE and MAE training set
#rmse_train_dr: 0.5125717724717413
#mae: 0.38670948432903857

y_pred_valid_pipe = pipeline.predict(dfm2v)
print(f"rmse_validation_dr: {np.sqrt(mean_squared_error(y_validation,y_pred_valid_pipe))}")
print(f"mae_validation_dr: {mean_absolute_error(y_validation,y_pred_valid_pipe)}")

##RMSE and MAE validation
#rmse_validation_dr: 0.6513396672387772
#mae_validation_dr: 0.5065934231826608

##model evaluation
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
param= dict()
param['max_depth'] =[80, 90, 100,None]
param['max_features'] = ["sqrt", "log2",None]
param['min_samples_leaf']= [1, 3, 5,7]
param['min_samples_split']= [2, 3, 4,5,7]
param['criterion']=["mse"]
param['splitter']= ["best", "random"]


cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)

grid_search = GridSearchCV(model,param_grid = param, cv = cv,scoring='neg_mean_absolute_error',  n_jobs = -1)
grid_search.fit(dfm2, y_train)

print('Best Score: %s' % grid_search.best_score_)
print('Best Hyperparameters: %s' % grid_search.best_params_)

y_pred_dr_gs = grid_search.predict(dfm2)
y_pred_valid_gs = grid_search.predict(dfm2v)

print(f"rmse_train_dr_gs: {np.sqrt(mean_squared_error(y_train,y_pred_dr_gs))}")
print(f"mae_train_dr_gs: {mean_absolute_error(y_train,y_pred_dr_gs)}")

#mse_train_dr_gs: 0.5314460799989538
#mae_train_dr_gs: 0.41282995893085683

print(f"rmse_valid_dr_gs: {np.sqrt(mean_squared_error(y_validation,y_pred_valid_gs))}")
print(f"mae_valid_dr_gs: {mean_absolute_error(y_validation,y_pred_valid_gs)}")

##
#rmse_valid_dr_gs: 0.5972800053694122
#mae_valid_dr_gs: 0.46584810571968754

##results from 10 neighbors knn imputation , try to get better results
##from hyperparameter optimization










