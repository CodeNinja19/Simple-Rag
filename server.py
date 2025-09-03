from fastapi import FastAPI
from RagUtil import call

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/call")
async def call_endpoint():
    response = await call("Which institute data you have context on ?")
    return {"response":response}