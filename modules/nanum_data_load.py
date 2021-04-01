import os
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

font_path = 'C:\\Users\\jmani\\AppData\\Local\\Microsoft\\Windows\\Fonts\\'
fonts = os.listdir(font_path)
fonts = [i for i in fonts if "나눔손글씨" in i]
fonts = np.array(fonts)

Syllables = ['030', '031', '032', '033', '034', '035', '036', '037', '038', '039']
Syllables = np.array(Syllables)

unicodeChars = chr(int(Syllables[0], 16))

plt.figure(figsize=(15, 15))

for uni in tqdm(Syllables):

    unicodeChars = chr(int(uni, 16))

    path = "./ttt/" + unicodeChars

    os.makedirs(path, exist_ok=True)

    for ttf in fonts:
        font = ImageFont.truetype(font=font_path + ttf, size=100)

        x, y = font.getsize(unicodeChars)

        theImage = Image.new('RGB', (x + 5, y + 3), color='white')

        theDrawPad = ImageDraw.Draw(theImage)

        theDrawPad.text((0.0, 0.0), unicodeChars[0], font=font, fill='black')

        msg = path + "/" + ttf[:-4] + "_" + unicodeChars

        theImage.save('{}.png'.format(msg))
