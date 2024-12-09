import os
from PIL import Image

import torch
from torchvision import transforms

import models


if __name__ == '__main__':
    # load pretrained model
    model = models.make(torch.load('app/weights/swinir-lte.pth')['model'], load_sd=True).cuda()
    # load images in folder path
    image_path = 'data/input/urban_img'
    image_list = os.listdir(image_path)
    for image in image_list:
        image_name = os.path.basename(image)[:-4]
        img = transforms.ToTensor()(Image.open(os.path.join(image_path, image)).convert('RGB'))
        img = img.unsqueeze(0).cuda()
        with torch.no_grad():
            output = model.gen_feat(img)
        # 텍스트 파일에 urban 이미지의 이름, h, w 저장
        with open('data/input/urban_hw.txt', 'a') as f:
            f.write(image_name + ' ' + str(output.shape[2]) + ' ' + str(output.shape[3]) + '\n')
        torch.save(output, 'data/input/urban_feat/' + image_name + '_feat.pt')
        torch.save(model.coeff, 'data/input/urban_feat/' + image_name + '_coef.pt')
        torch.save(model.freqq, 'data/input/urban_feat/' + image_name + '_freq.pt')