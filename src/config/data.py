from pydantic_settings import BaseSettings
from typing import List


class DataSettings(BaseSettings):
    wikiart_dir: str = "data/wikiart_dir"
    artemis_dir: str = "data/artemis"
    emotion_list: List[str] = ["something else", "sadness", "contentment", "awe", "amusement", "excitement", "fear", "disgust", "anger"]


    class Config: 
        env_file = "config/.env"


data_settings = DataSettings()