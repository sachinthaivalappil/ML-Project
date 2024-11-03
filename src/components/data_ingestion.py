from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd
import os 
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
import sys


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts',"train.csv")
    test_data_path: str = os.path.join('artifacts',"test.csv")
    raw_data_path: str = os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.Ingestion_config = DataIngestionConfig

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion method")
        try:
            df=pd.read_csv('/Users/sachints/Desktop/projects/ML Project/ML-Project/notebook/data/stud.csv')
            logging.info("Data Read completed successfully")
            os.makedirs(os.path.dirname(self.Ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.Ingestion_config.raw_data_path,index=False,header=True)
            logging.info("train test initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.Ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.Ingestion_config.test_data_path,index=False,header=True)
            logging.info("Data ingestion completed successfully")

            return (self.Ingestion_config.test_data_path,self.Ingestion_config.train_data_path)
        except Exception as e: 
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    dataingestion= DataIngestion()
    test_path,train_path=dataingestion.initiate_data_ingestion()
    data_transform= DataTransformation()
    data_transform.initiate_data_transformation(train_path,test_path)
        
