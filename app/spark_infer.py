from app.utils import make_coord
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType

import torch
from torchvision.transforms import ToTensor
from PIL import Image
import os

import models

upscale = 4
batch = 1

image_path = 'img_001.png'

# Spark 세션 생성
spark = SparkSession.builder \
    .appName("Distributed-LTE") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

model = models.make(torch.load('app/weights/swinir-lte.pth')['model'], load_sd=True).cuda()
model.eval()
broadcast_model = spark.sparkContext.broadcast(model)

broadcast_feat = spark.sparkContext.broadcast(torch.load('data/input/urban_feat/img_001_feat.pt'))
broadcast_coef = spark.sparkContext.broadcast(torch.load('data/input/urban_feat/img_001_coef.pt'))
broadcast_freq = spark.sparkContext.broadcast(torch.load('data/input/urban_feat/img_001_freq.pt'))

h, w = broadcast_feat.value.shape[2], broadcast_feat.value.shape[3]
feat_coord = make_coord((h, w), flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(1, 2, h, w)
output_coord = make_coord((h * upscale, w * upscale))

cell = torch.ones_like(output_coord)
cell[:, 0] *= 2 / (h * upscale)
cell[:, 1] *= 2 / (w * upscale)

def split_coords(output_coord, batch):
    coords = output_coord.view(-1, 2).tolist()
    for i in range(0, len(coords), batch):
        yield coords[i:i + batch, :]

data = []
for batch in split_coords(output_coord, batch):
    data.append({
        "image_path": image_path,
        "output_coord": batch,
        "feat_coord": feat_coord,
        "cell": cell
    })

@udf(returnType=ArrayType(FloatType()))
def process_image(image_name, output_coord, feat_coord, cell):
    model = broadcast_model.value
    model.feat = broadcast_feat.value
    model.coef = broadcast_coef.value
    model.freq = broadcast_freq.value

    with torch.no_grad():
        output = model.query_rgb(output_coord, cell)
    return output.squeeze(0).tolist()


df = spark.createDataFrame(data)
df = df.withColumn("result", process_image(df.image_path))

df.write.csv("/output/results.csv", header=True)