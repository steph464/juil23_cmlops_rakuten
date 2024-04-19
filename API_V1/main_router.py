from fastapi import FastAPI
from user.user_router import router as user_router
from predict.predict_router import predict_router as predict_router

api = FastAPI(
    openapi_tags=[
        {'name': 'Default'},
        {'name': 'Users', 'description': 'Admin functionalities'},
        {'name': 'bonus', 'description': 'route to add as needed'},
        {'name': 'Predictions', 'description': 'NLP, Computer Vision'},
    ],
    title="API - Rakuten France Multimodal Product Data Classification",
    description="""Text / image classification powered by FastAPI.

Login: authentication required

To use, select:

1: : POST)
2: : POST)
3: xxxxxx  """,
    version="0.0.1",
    contact={"name": "Your Name", "email": "your@email.com"},
)

# Mount user router
api.include_router(user_router, prefix="/Users", tags=["Users"])
api.include_router(predict_router, prefix="/Predictions", tags=["Predictions"])














