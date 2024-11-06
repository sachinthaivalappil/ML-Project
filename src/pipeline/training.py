from src.components.data_ingestion import DataIngestion 
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging



class Model_buildingPipeline:
    def __init__(self):
        pass

    def main(self):
        logging.info("Data ingestion starting")
        dataingestion= DataIngestion()
        test_path,train_path=dataingestion.initiate_data_ingestion()
        data_transform= DataTransformation()
        train_arr,test_arr,_=data_transform.initiate_data_transformation(train_path,test_path)
        model_train= ModelTrainer()
        print(model_train.initiate_model_trainer(train_arr,test_arr))


    