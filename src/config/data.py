from pydantic_settings import BaseSettings
from typing import List


class DataSettings(BaseSettings):
    wikiart_dir: str
    artemis_dir: str
    emotion_list: List[str]


    class Config: 
        env_file = "config/.env"


data_settings = DataSettings()