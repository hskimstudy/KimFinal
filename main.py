from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import joblib
import pandas as pd
import shap
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #이거 나중에 수정해야함 보안보안
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)



# 모델 로드
with open('lgb_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
explainer = shap.TreeExplainer(model)

merged_data = pd.read_csv('merged_data.csv')
movie_info = pd.read_csv('movie_info.csv')

# 특성 목록 정의
features = ['age', 'occupation', 'Action', 'Adventure', 'Animation', "Children's",
    'Comedy', 'Crime', 'Documentary', 'Drama',
    'Fantasy', 'Film-Noir', 'Horror', 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
    'War', 'Western', 'gender']

# FastAPI에서 사용할 입력 데이터 모델 정의
class UserInput(BaseModel):
    age: int
    occupation: int
    gender: str
    genres: dict
  

# API 엔드포인트 생성
@app.post("/predict")
async def predict_rating(user_data: UserInput):
    try:
        # User Input 데이터를 모델 입력 형식으로 변환

        genre_columns = ['Action', 'Adventure', 'Animation', "Children's",
            'Comedy', 'Crime', 'Documentary', 'Drama',
            'Fantasy', 'Film-Noir', 'Horror', 'Musical',
            'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
            'War', 'Western']
        features = ['age', 'occupation'] + genre_columns + ['gender']

        user_features = [user_data.age, user_data.occupation] + [user_data.genres.get(genre, 0) for genre in genre_columns] + [1 if user_data.gender == 'M' else 0]
        
        user_df = pd.DataFrame([user_features], columns=features)

        # 모델로 예측 수행
        user_pred = model.predict(user_df)[0]

        # 예측 결과 반환
        # Get recommended movies with average ratings higher than user's predicted rating
        unique_movies = merged_data.drop_duplicates(subset=['movie_id'])
        average_ratings = unique_movies.groupby('movie_id')['rating'].mean().reset_index()

        recommended_movies = average_ratings[average_ratings['rating'] >= user_pred]
        recommended_movies = recommended_movies.sort_values(by='rating', ascending=False).head(6)

    # Merge with movie titles
        recommended_movies = recommended_movies.merge(movie_info[['movie_id', 'title']], on='movie_id', how='inner')

    # Calculate SHAP values for the individual user input
        shap_values_user = explainer.shap_values(user_df)

    # Separate positive and negative SHAP values
        positive_shap_values = shap_values_user[0].copy()
        positive_shap_values[positive_shap_values < 0] = 0
        negative_shap_values = shap_values_user[0].copy()
        negative_shap_values[negative_shap_values > 0] = 0

    # Sort the positive and negative SHAP values separately
        sorted_positive_indices = np.argsort(positive_shap_values)[::-1]
        sorted_negative_indices = np.argsort(negative_shap_values)

    # Get the names of the top 3 positive and bottom 3 negative features
        top_positive_features = [features[i] for i in sorted_positive_indices[:3]]
        top_negative_features = [features[i] for i in sorted_negative_indices[:3]]

        return {
        "predicted_rating": user_pred,
        "recommended_movies": recommended_movies[['title', 'rating']].to_dict(orient='records'),
        "shap_values": shap_values_user[0].tolist(),
        "shap_values_positive": [
        {
            "feature": feature,
            "shap_value": positive_shap_values[features.index(feature)]
        } for feature in top_positive_features
    ],
    "shap_values_negative": [
        {
            "feature": feature,
            "shap_value": negative_shap_values[features.index(feature)]
        } for feature in top_negative_features
    ]
    }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
