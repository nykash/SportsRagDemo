from fastapi import FastAPI
from apis import upload  # import other routers here

app = FastAPI(title="Volleyball Clip Finder", version="1.0.0")

# Include routers
app.include_router(upload.router)

@app.get("/")
async def root():
    return {"message": "Welcome to Volleyball Clip Finder"}