
from fastapi import APIRouter, FastAPI, HTTPException,Request,Body
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import os,re,logging,urllib.parse,uvicorn
from starlette.middleware.cors import CORSMiddleware
from src.pipeline.prediction import CustomData,PredictPipline
from pydantic import BaseModel
from src.exception import CustomException
import sys
from src.logger import logging
from src.pipeline.training import Model_buildingPipeline


logging.getLogger().setLevel(logging.DEBUG)

routes = APIRouter()

class PredictRequest(BaseModel):
    gender: str
    race_ethnicity : str
    parental_level_of_education: str
    lunch:str 
    test_prepration_course:str
    reading_score:int
    writing_score: int


  
@routes.post("/predict")
async def prediction(data: PredictRequest = Body(...)):
    try:
        custom_data=CustomData( gender= data.gender,
        race_ethnicity =data.race_ethnicity,
        parental_level_of_education=data.parental_level_of_education,
        lunch=data.lunch, 
        test_prepration_course=data.test_prepration_course,
        reading_score=data.reading_score,
        writing_score= data.writing_score)
        data=custom_data.get_data_as_data_frame()
        predict=PredictPipline()
        results=predict.predict(data)
        results=results[0]
        return JSONResponse(content=results, status_code=200)
    except Exception as e:
        raise CustomException(e,sys)

@routes.post("/training")
async def training():
    STAGE_NAME = "Data Ingestion and Training"
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
        data_validation = Model_buildingPipeline()
        data_validation.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        return {"message": "Training completed successfully."}
    except Exception as e:
            raise CustomException(e,sys)



    






# Add the custom middleware to the FastAPI app

def init_app() -> FastAPI:
    app = FastAPI()
    app.include_router(routes)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust this as necessary for your deployment
        allow_credentials=True,
        allow_methods=["*"],
    )
    return app
if __name__ == "__main__":
    PORT = int(os.getenv("PORT", default=5050))
    HOST = os.getenv("HOST", default="0.0.0.0")
    app = init_app()
    if app is None:
        raise TypeError("app not instantiated")
    uvicorn.run(app, host=HOST, port=PORT)
