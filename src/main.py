# python main.py --img_file ../data/line5.png
from typing import Tuple, List
from fastapi import FastAPI,File, UploadFile
import shutil
from pydantic import BaseModel

import cv2
from path import Path

from dataloader_iam import DataLoaderIAM, Batch
from model import Model, DecoderType
from preprocessor import Preprocessor

app = FastAPI()


def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    if line_mode:
        return 256, 32
    return 128, 32


def char_list_from_file() -> List[str]:
    with open('../model/charList.txt') as f:
        return list(f.read())



def infer(model: Model, fn_img: Path) -> None:
    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    (thresh, thres_img) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    assert img is not None

    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)
    thres_img = preprocessor.process_img(thres_img)

    batch = Batch([img], None, 1)
    batch1 = Batch([thres_img], None, 1)

    recognized = model.infer_batch(batch, True)
    recognized1 = model.infer_batch(batch1, True)

    return([recognized[0],recognized1[0]])

# Model instance creation
decoder_mapping = {'bestpath': DecoderType.BestPath,'beamsearch': DecoderType.BeamSearch}
decoder_type = decoder_mapping['bestpath']
model = Model(char_list_from_file(), decoder_type, must_restore=True)
class request_body(BaseModel): 
    filename : str


@app.post('/test')
def test():
    return{'Hello':'Hi'}

@app.post('/predict')
def predict(data: request_body):
    ans = infer(model,'../data/'+ data.filename)
    return{ "Without Thresholding": ans[0], "With Thresholding":ans[1] }

@app.post("/upload")
def create_upload_file(file: UploadFile = File(...)):
    with open('../data/'+file.filename, "wb") as buffer:
      shutil.copyfileobj(file.file, buffer)
    
    ans = infer(model,'../data/'+ file.filename)
    print(ans)
    return { "Without Thresholding": ans[0], "With Thresholding":ans[1] }