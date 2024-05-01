import librosa
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import argparse

def getDataFrame(path, start, end):
    fileNames = os.listdir(path=path)
    fileNames = fileNames[start:end]
    for fileName in tqdm(fileNames):
        if 'wav' in fileName:
            patFileName = os.path.join(path, fileName)
            y, sr = librosa.load(patFileName, sr=16000)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            img = librosa.display.specshow(mfcc, sr=sr)
            plt.axis('off')
            plt.savefig(f"{path}/converted/{fileName}.png", bbox_inches='tight', pad_inches=0)


parser = argparse.ArgumentParser()
parser.add_argument('-path')
parser.add_argument('-start')
parser.add_argument('-end')
args = parser.parse_args()
getDataFrame(args.path, eval(args.start), eval(args.end))
