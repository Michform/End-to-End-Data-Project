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

			model = {"CatBoostRegressor":CatBoostRegressor(),
				"RandomForestRegressor":RandomForestRegressor(),
				"GradientBoostingRegressor":GradientBoostingRegressor(),
				"AdaBoostRegressor":AdaBoostRegressor(),
				"LinearRegression":LinearRegression(),
				"DecisionTreeRegressor":DecisionTreeRegressor(),
				"XGBRegressor":XGBRegressor(),
				"KNeighborsRegressor":KNeighborsRegressor(),
				"SVR":SVR()
            }

			model_report : dict =evaluate_model(X_train=X_train, 
                                       y_train=y_train, 
                                       X_test=X_test, 
                                       y_test=y_test, 
                                       models=model)
   
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
