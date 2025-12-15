from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import numpy as np
import cv2
import os

from color_core import auto_color_correct

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ==============================
# Главная страница
# ==============================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# ==============================
# ИИ-обработка изображения
# ==============================
@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    style: str = Form("neutral"),
    intensity: int = Form(80)
):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    result = auto_color_correct(img, style=style, intensity=intensity)

    result_path = "static/result.jpg"
    cv2.imwrite(result_path, result)

    return JSONResponse({
        "result_image": "/" + result_path
    })
