import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
	GradientBoostingRegressor,
	AdaBoostRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
	model_trainer_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
	def __init__(self):
		self.model_trainer_config = ModelTrainerConfig()

	def initiate_model_trainer(self, train_array, test_array):
		try:
			logging.info("Split training and test input data")
			X_train, y_train, X_test, y_test = (
				train_array[:, :-1],
				train_array[:, -1],
				test_array[:, :-1],
				test_array[:, -1],
			)

			model = {"CatBoostingRegressor":CatBoostRegressor(),
				"RandomForestRegressor":RandomForestRegressor(),
				"GradientBoostingRegressor":GradientBoostingRegressor(),
				"AdaBoostRegressor":AdaBoostRegressor(),
				"LinearRegression":LinearRegression(),
				"DecisionTreeRegressor":DecisionTreeRegressor(),
				"XGBRegressor":XGBRegressor(),
				"KNeighborsRegressor":KNeighborsRegressor(),
				"SVR":SVR()
            }
			
   
			params={
                "DecisionTreeRegressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForestRegressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoostingRegressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "LinearRegression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoostingRegressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "KNeighborsRegressor":{
					'n_neighbors': [5, 7, 9, 11],
					'weights': ['uniform', 'distance']
				},
				"SVR":{
					'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
					'C': [0.1, 1, 10, 100],
					'gamma': ['scale', 'auto']
				}
                
            }


			model_report : dict =evaluate_model(X_train=X_train, 
                                       y_train=y_train, 
                                       X_test=X_test, 
                                       y_test=y_test, 
                                       models=model,
                                       param=params)
   
			best_model_score = max(sorted(model_report.values()))
   
			best_model_name = list(model_report.keys())[
       			list(model_report.values()).index(best_model_score)]
   
			best_model = model[best_model_name]
			if best_model_score < 0.6:
				raise CustomException("No best model found", sys)	

			logging.info(f"Best model found on both training and testing dataset: {best_model_name} with r2 score: {best_model_score}")
  
			save_object(
				file_path=self.model_trainer_config.model_trainer_file_path,
				obj=best_model
			)	
  
			prediction = best_model.predict(X_test)
			r2_square = r2_score(y_test, prediction)
			return r2_square	

		except Exception as e:
			raise CustomException(e, sys)
