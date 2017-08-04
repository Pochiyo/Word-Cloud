
import jieba
import numpy
import codecs
import pandas
import matplotlib.pyplot as plt
from scipy.misc import imread
from wordcloud import WordCloud, ImageColorGenerator

file = codecs.open("machine.txt", 'r', 'gbk')
content = file.read()
file.close()

jieba.load_userdict('dict_machine.txt');

segments = []
segs = jieba.cut(content)
for seg in segs:
    if len(seg)>1:
        segments.append(seg);
segmentDF = pandas.DataFrame({'segment':segments})

stopwords = pandas.read_csv(
 "StopwordsCN.txt", 
 encoding='gbk', 
 index_col=False,
 quoting=3,
 sep="\t"
 )
  
wyStopWords = pandas.Series([ 
    'A', 'B', 'C', 'D', 'E', 'E', 'G', 'H', 'I', 'J', 'K', 
   'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
   'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', 
   '8', '9', '0', '12', '2015','2016','09','11','DX380LC','15','07','9C','WL53'
   ' ', ''
 ]);
 
segmentDF = segmentDF[~segmentDF.segment.isin(wyStopWords)]
    

segStat = segmentDF.groupby(
by=["segment"]
)["segment"].agg({
"jishu":numpy.size
}).reset_index().sort(
columns=["jishu"],
ascending=False
);
segStat.head(100)
# print(segStat)
segStat1=segStat.drop(0)
print(segStat1)

bimg = imread("image.jpg")
wordcloud = WordCloud(
background_color="white", 
mask=bimg, font_path='yahei.ttf'  
 )

wordcloud = wordcloud.fit_words(segStat1.head(1000).itertuples(index=False))

bimgColors = ImageColorGenerator(bimg)

plt.axis("off")
plt.imshow(wordcloud.recolor(color_func=bimgColors))
plt.show()