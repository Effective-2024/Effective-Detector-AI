import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models.conv3d_m import Conv3dM
from fastapi import FastAPI, UploadFile, File
import torchvision.transforms as transforms
import io
from PIL import Image
import numpy as np
from fastapi.responses import JSONResponse
from typing import List

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = Conv3dM()
model.load_state_dict(torch.load('/home/insung/ai_server/weights/conv3d_m.pt'))
model.eval()
model.to(device)

app = FastAPI()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),        
])

@app.post("/inference")
async def inference(files: List[UploadFile] = File(...)):
    frames = []
    for file in files:
        print(f"Received file: {file.filename}")
        frame = Image.open(io.BytesIO(await file.read()))
        frames.append(transform(frame))


    frames = torch.stack(frames)  
    frames = frames.unsqueeze(0)  
    frames = frames.permute(0, 2, 1, 3, 4)  

    with torch.no_grad():
        frames = frames.to(device)
        logit = model(frames)
        pred = logit.argmax(1).detach().cpu().numpy()
    
    print(f"pred = {pred}")

    
    
    #return pred
    return JSONResponse({"pred": pred.tolist()})




