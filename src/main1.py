from typing import Tuple, List
from pydantic import BaseModel
import cv2
from path import Path
from dataloader_iam import Batch
from model import Model, DecoderType
from preprocessor import Preprocessor

class FilePaths:
    fn_char_list = '../model/charList.txt'
    fn_summary = '../model/summary.json'
    fn_corpus = '../data/corpus.txt'

def get_img_height() -> int:
    return 32

def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()

def char_list_from_file() -> List[str]:
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())

def infer(model: Model, fn_img: Path) -> None:
    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    (thresh, thres_img) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("Original Image",img)
    cv2.imshow("Binary Image",thres_img)
    
    assert img is not None

    preprocessor = Preprocessor(get_img_size(True), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)
    thres_img = preprocessor.process_img(thres_img)

    batch = Batch([img], None, 1)
    batch1 = Batch([thres_img], None, 1)

    recognized = model.infer_batch(batch, True)
    recognized1 = model.infer_batch(batch1, True)

    print(f'Without Thresholding: "{recognized[0]}"')
    print(f'With Thresholding: "{recognized1[0]}"')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
decoder_mapping = {'bestpath': DecoderType.BestPath,'beamsearch': DecoderType.BeamSearch}
decoder_type = decoder_mapping['bestpath']
model = Model(char_list_from_file(), decoder_type, must_restore=True)

file = input('Enter the File name: ')
infer(model,'../data/'+ file)