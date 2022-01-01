import shutil
from fastapi import FastAPI, Request
from fastapi import File, UploadFile, Form
import uvicorn
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from fastapi.responses import RedirectResponse

app = FastAPI()
app.mount("/static/", StaticFiles(directory="static"), name="static")


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
    # Use bvh function

    gif_src = "test.gif"
    filename = file.filename

    return templates.TemplateResponse("index.html", {"request": request, "gif_src": gif_src, "filename": filename})


@app.get("/download/{filename}")
async def create_upload_file(request: Request, filename):
    file_location = "uploaded_files/"+filename
    print("Hi")
    print(file_location)
    return FileResponse(file_location, media_type='application/octet-stream')




if __name__ == "__main__":
    uvicorn.run(app, port=8000)
