from config.data import data_settings

emotion_to_index = {emotion: i for i, emotion in enumerate(data_settings.emotion_list)}
index_to_emotion = {v: k for k, v in emotion_to_index.items()}