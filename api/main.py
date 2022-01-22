import os.path
import shutil
from fastapi import FastAPI, Request
from fastapi import File, UploadFile, Form
import uvicorn
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from fastapi.responses import RedirectResponse
from src.utils import get_project_path
from api.model.inference import InferModel
ROOT_PATH=get_project_path()
app = FastAPI()
app.mount("/static/", StaticFiles(directory="static"), name="static")
app.mount("/results/", StaticFiles(directory="results"), name="results")
inference_model=InferModel()
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def index(request: Request):
    gif_src = "img.png"
    filename = ""
    return templates.TemplateResponse("index.html", {"request": request, "gif_src": gif_src, "filename": filename})


@app.post("/generate/")
async def create_upload_file(request: Request, file: UploadFile = File(...)):
    file_path = "uploaded_files/" + file.filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file.file.close()

    except:
        return RedirectResponse("http://127.0.0.1:8000/", status_code=303)

    # Use gif function
    print(file_path)
    inference_model.infer(file_path=file_path)
    # Use bvh function

    gif_src = "animation.gif"
    filename = 'test_013.bvh'


    return templates.TemplateResponse("index.html", {"request": request, "gif_src": gif_src, "filename": filename})


@app.get("/download/{filename}")
async def create_upload_file(request: Request, filename):
    file_location = os.path.join(ROOT_PATH,"api\\results\\bvh",filename)
    print("Hi")
    print(file_location)
    return FileResponse(file_location, media_type='application/octet-stream')




if __name__ == "__main__":
    uvicorn.run(app, port=8000)
