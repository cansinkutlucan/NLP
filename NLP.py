# %% NLP
# 8 session - 1 lab - 1 PROJECT(ML,DL,NLP karışımı)

# %% NLP-1
# NLP : Natural Language Processing
# Insan dilinin bazı algoritma, model vs tarafından anlaşılıp kullanıldığı yöntemdir
# CNN ve NLP özel olarak ayrı bir şekilde uzmanlaşılacak alanlardır
# Çoğunlukla classification üzerinde yoğunlaşacağız. Diğer uygulamaları hazır paketler üzerinden göstereceğiz
# Junior seviyede pozisyonlara(NLP için) hazır olacaksınız

# SofiaH: NLP: ML, DL ve AI alanlarının hepsinden faydalanır. Text'lerin makinenin anlayabileceği forma dönüştürülmesidir

# ML, DL de kullandığımız datalar nasıl datalardı. Tabular(Satır ve sütunlardan oluşan / Yapılandırılmış) datalardı(CNN hariç)
# Ancak her türlü datayı satır ve sütunlara dönüştürmemiz kolay değil
# NLP ile datamızı yapılandırılmış dataya dönüştürmeden yazı üzerinde bizim istediğimiz keyword ler üzerine yoğunlaşıp classification veya
# .. sentiment analysis yapabiliyoruz

# NLP kullanım alanları
# Diagnosing(Healthcare), Sentiment Analysis, Translator, ChatBot, Classifying emails, Detecting Fake News, Recruiting Assistant,
# .. Intelligent Voice-Driven Interfaces, Litigation(Dava) Tasks(To automate routine litigation tasks and help courts save time)
# NOT: (johnsnowlabs firmasında) bir araştırmada doktorlar %60-65 civarı doğru teşhis yaparken makineler %80 üzerinde doğru tahmin yapmışlar
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, GRU, Embedding, Dropout
from tensorflow.keras.models import Sequential
from gensim.models import KeyedVectors
import zipfile
from gensim.models import Word2Vec
from wordcloud import WordCloud
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from yellowbrick.classifier import PrecisionRecallCurve
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB, BernoulliNB # BernoulliNB for binary model
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import warnings
import numpy as np
import pandas as pd
# Diagnosing(Healthcare)
# SofiaH: Hastaların tedavi süresince yapılan işlemler modele verilir. Model text verilerine veya labdan gelen sayısal verilere
# .. dayanarak hastalık analizi yapar.
# Sentiment Analysis
# SofiaH: Firmalar, müşterilerin yaptıkları yorumlar üzerinden kendini geliştirmeye çalışıyor. Hükümetler de kanun çıkarırken yapılan
# .. yorumlara göre hareket ediyorlar
# ChatBot
# Machine Translation
# SofiaH: Google translate bunu kullanıyor
# Digital Assistant(Speech Recognation)
# SofiaH: Telefon şirketlerini aradığımızda karşımıza çıkan digital aşistan. Konuşmayı "speech recognition" programı
# Verilen sesli komutları text e çevirip bunlar anlamlandırılıp ona göre makine bir anlam çıkarıyor
# Ancak çok başarılı değiller. Mesela tel paketinizi değiştirmek istediğinizde sürekli tekrar eden cevaplar veriyor
# Google ın geliştirdiği digital assistant-3 yıllık teknoloji
# Google ın eğittiği model, bir kuaförü arayacak ve randevu almaya çalışacak alttaki linkte
# https://www.youtube.com/watch?v=D5VN56jQMWM

# Bunları yapabilmek için çok büyük datalara sahip olmak lazım(Google, Microsoft vs gibi firmalar)(Gmail de atılan mailler, çeviriler vs vs))
# Örneğin; Google translate de günlük 13 milyar çeviri yapılıyor. Ortalama 2 milyarı yeni kullanılan cümleler
# Bizim eğittiğimiz modeller bazen yetersiz kalıyor, google gibi firmaların eğittiği hazır modelleri kullanacağız bu durumlarda

# NLP teori kısmı ve terimler
# NOT: Piyasa da en çok kullanılan sentiment analysis ve classification a bakacağız burada
# NOT: Diğer uygulamaları hazır modeller üzerinden anlatacağız
# Corpora  : Bir çok corpus un bir arada olduğu alan. Buna NLP özelinde corpora denir. Database diyebiliriz
# Corpus   : Data seti.
# Document : Corpustaki her bir satır. Bir kelimeden, bir cümleden, bir paragraftan, bir kaç sayfadan, bir kitaptan oluşabilir

"""
# SofiaH
1. Tokenization and lower
2. Remove punctuation, special characters and numbers
3. Remove stopwords
4. Data Normalization (Lemmatization-Stemming)
5. a.Count Vectorization b.TF-IDF
"""

# Tokenization
# Data cleaning in 1. aşaması Tokenization dır
# SofiaH: Elimizdeki mevcut text in makinalar tarafından daha kolay anlaşılabilmesi için daha küçük parçalara bölünmesidir
# 2 farklı tokenization işlemi var. 1. Sentence 2.Word
# Çoğunlukla word tokenization kullanacağız. Sentence tokenization ın piyasa da çok kullanımı yok
# Slayttaki cümlede cümleyi ayrı kelimelere(tokenlere) ayırmışız.(Tokenleştirme yapılmış)
# Cümledeki kelimeleri numaralandırabilirim --> Örneğin: This is a sample
# This   --> 1 numara # is     --> 2 numara # a      --> 3 numara # sample --> 4 numara
# Tokenlerimiz numaralandırıp bu tokenler kendi arasında anlamsal ilişki kurabilir. Cümleler halinde verirsek model daha zor öğrenir
# Bunları sayısal hala dönüştürüp modele vereceğiz ve model bunlardan anlam çıkaracak
# Cümle olarak verirsek modele, modelin bunu anlamlandırması daha zor olacaktır. Bunun yerine belli başlı kelimeler
# .. kullanayım bu kelimelerle cümle oluşturma ve makina için anlaşılması daha kolay olacaktır.
# SofiaH: Datadaki kelimelerin hepsi modele saydırılır. Saydırma işlemi yapılmazsa CountVectorizer, TF-IDF Vectors işlemleri de yapılamaz

# Lower
# Data Cleaning in 2. aşaması: Tokenlerimizi küçük harflere dönüştürmezsem model hepsine farklı token muamelesi
# ..  yapar(THIS, This, this --> Bu 3 ünü farklı 3 token olarak algılar)
# .. bu da modelimizin öğrenememesine(ya da yanlış öğrenmesine) yol açar.
# Bu 2 aşamayı(Tokenization, Lower) ML ve DL modellerinde kesinlikle yapmalıyız. Bundan sonraki aşamaları DL de ister yapın ister yapmayın
# .. noktalama işaretlerinden ayıklamayı da DL de kaldırmasanız da model bunları anlamlandırabiliyor ama maliyet olur

# Remove Puncs and numbers
# Cleaning in 3. aşaması: Noktalama işaretlerin temizlenmesi
# Keyword leri aradığı için model bunları gürültü olarak görüyor
# Zaten 10000-15000 tane feature ımız olacak. Bunları da eklemeye gerek yok
# NOT: SofiaH: Classification ve sentimental analizlerde sayısal verilerde temizlenir

# Remove stopwords
# Cleaning in 4. aşaması: stopwords leri çıkarma
# Stopwords: Datamızın anlamına extra bir derinlik katmayan kelimeler: bağlaçlar, sıklık zarfları, soru kelimeleri vs vs
# Örneğin: Onlar beğendiler, Onlar beğenmediler --> Burada 1 olumlu 1 olumsuz cümle olmasına karşın
# .. "onlar" kelimesi olumlu ya da olumsuz bir anlam vermiyor o yüzden bu kelimeyi boş yere almıyoruz(feature olarak)
# nltk kütüphanesini kullanacağız stopwords için(179 tane bizim işimizi görecek). Bunları datanızda tutsanızda classification ve sentiment
# .. analizini güzel yapar ama hesaplama maliyetiniz olur.

# Normalization
# Cleaning 5. aşama: Tokenlerin köklerine inme --> diğer bir ad ile "Normalization"
# ML de bunu yapacağız. DL de tokenlerin asıl halleriyle bulunması lazım. Çünkü DL de anlamsal ilişki, olduğu halleriyle kurulabilir
# "beğendin" mi "beğenmedin" mi bunu araştırıyoruz  . Beğendim, beğendin, beğenildi, beğenilmedi ..... vs vs
# Farklı varyasyonlarını iptal edip kelimenin kökenine inmek --> "beğenmek"
# 2 farklı yöntem vardır
# Lemmatization: Bu tokenin kökenine inersem anlam kaybı olur mu olmaz mı bakar(Arabacı, dondurmacı --> araba, dondurma --> anlam değişti)
# .. anlam kaybı olduğunu görürse ekleri atmaz "Lemmatization". Best practice olarak Lemmatization kullanılır ama
# .. Lemmatization kullanmak zorunlu değildir ama tavsiyemiz Lemmatization kullanmanızdır
# Stemming : Direk kökleri alır. Anlam değişiyormuş, değişmiyormuş bakmaz.
# .. Stemming bazen saçmalar. Johnson hoca: Mesela "koyun" kelimesine "koy" dediğini gördüm

# NOT: SofiaH: Model, stopword'ler içindeki olumlu-olumsuz yardımcı fiillerle(should, couldn't) sentimental analysis yapacağı
# .. zaman ilgilenir. Bu tarz modellerde bu stopwordler atılMAMALIDIR.
# NOT: SofiaH: Stopword lerin kullanılan kütüphaneler : NLTK(179 English Stopwords), spaCy ve gensim 

# Örnek: Classification dan ziyade alttaki örneği sentiment analizi(olumlu-olumsuz) şeklinde değerlendirmek daha iyi
# Sample_text = "Oh man, this is pretty cool. We will do more such things."
# Tokenization : ['oh','man',',','this', 'is','pretty','cool','.','we','will','do','more','such','things','.']
# Removing punctation: ['oh','man','this','is','pretty','cool','we','will','do','more','such','things']
# Removing stopwords : ['oh','man','pretty','cool','things']
# lemmatization : ['oh','man','pretty','cool','thing'] # # stemming : ['oh','man','pretti','cool','thing']

# Model "Pretty" ve "cool" vs gördüğünde büyük ihtimalle pozitif bir sonuç döndürecek zaten. Olayın ana fikrine bakıyoruz yani
# .. Bu cümlenin tamamını hatırlamayacağız ama bu cümlenin olumlu mu olumsuz mu olacağını bileceğiz. Model de bu şekilde çalışıyor
# "Pretty" nin "güzel" anlamının yanı sıra "oldukça" anlamı olmasına rağmen "pretti" şeklinde almış stemming de
# .. yani anlamı sadece "güzel" olan kelime köküne inilmiş.

# NOT: vectorizer.get_feature_names() : datamızda geçen bütün unique tokenleri tespit edip(daha sonra featurelara dönüştüreceğiz bunları)

# Vektorization
# Data temizleme işlemi bittikten sonra ham datayı modelin anlayabileceği şekilde sayısal forma çevirmemiz gerekiyor
# 1. Count vectorizer(countvectorizer) 2.TF-ITF(tfidfvectorizer) 3.Word Embedding(Word2Vec & Glove)
# Word embedding en advanced algoritmadır
# ilk 2 si ML modellerinde tercih edilir

# 1-Countvectorizer Yöntemi
# Corpusumuzdaki bütün yorumlarda geçen unique tokenleri elde ediyoruz önce sonra bunları herbirini stopwordlerden
# .. temizlenmiş şekilde alfabetik sıraya göre birer feature yaparak yazar
# SofiaH: CountVectorizer, keyword'lere öncelik tanırken, document da ne kadar sıklıkta geçtiğine dikkat eder
# .. Cümle içinde hangi tokenden kaçar tane var bunu sayar ve sayısı fazla olana fazla ağırlık verir
# SofiaH: CountVectorizer document içindeki hangi tokenin önemli olduğunu tespit edebilir fakat corpus içinde
# .. ne kadar önemli olduğunu tespit edemez.
# SofiaH: Modele olumlu ve olumsuz anlam katan token sayısı eşit ise CountVectorizer yorumun olumlu veya olumsuz olduğunu anlayamayabilir
# Document-1 : John likes to watch movies. Mary likes movies too
# Document-2 : Mary also likes to watch football games
# Corpusumuz: document-1 ve document-2 den oluşan kısım
# SofiaH: Alttaki örnekte CountVectorizer, "likes" ve "movies" e çok fazla ağırlık verecektir

"""
    also football games john likes mary movies to too watch
1 : 0       0       0    1     2     1   2     1   1   1
2 : 1       1       1    0     1     1   0     1   0   1
"""

# 2.TF-IDF Yöntemi
# Count vektorizer a göre daha gelişmiş bir modeldir. Ama bazen countvectorizer ile daha yüksek skorlar alabiliriz
# Countvektorizer da bir tokenin cümle içinde geçme sıklığına göre numaralandırma yapıyorduk
# .. Bir token hem corpus içerisinde hem de o yorum özelinde ne kadar kullanılmış insight ını vermiyordu count vektorizer.
# .. Bu insight ı TF-IDF veriyor. Yani;
# TF-IDF: Bir token hem corpus içerisinde hem de o yorum özelinde ne kadar kullanılmış insight ını bize vererek hesaplama yapar
# TF-IDF kullanacaksanız stopword leri silmeden devam edebilirsiniz ama yine de silerek devam etmeniz daha iyi
# .. çünkü TF-IDF ona göre sıklıkla geçen bu stopwordlerin katsayılarını küçük tutacak.
# .. Eğer bir tokenim bütün yorumlarda geçiyorsa o token üzerinden bir duygu analizi yapamazsanız anlamına geliyor

# TF : Bir tokenin yorum içerisindeki kullanım sıklığı( .. ile alakalı bir insight veriyor)
# "Ahmet Tülini beğendi" --> "beğendi" için --> TF = 1/3
# IDF: Bir tokenin corpus içerisindeki geçme sıklığı (... ile alakalı insight sağlıyor)
# "Inverse" kısmından önce, "Document frequency" yi açıklayalım
# 100 yorumum olsun. 1. yorumumda , 2. yorumumda ve 100. yorumumda "beğendi" kelimesi geçsin --> DF = 3 /100
# Inverse ü de ekleyelim. yani tersini --> 100/3  # NOT: Eksi değerden kurtulmak için Inverse işlemi yapıyoruz
# Sonra buna log ekleriz --> IDF = log(100/3)
# Peki neden log aldık(np.log(3/100) = -1.52 , np.log(100/3) = 1.52, --> np.log(1000000/3)=6.52)
# log bir nevi doğal scale yapıyor. Çalışma maliyetini azaltıp ağırlıklandırmayı ayarlıyor
# .. yani bazı featurelara fazla ağırlık verilip benim için önemli olan tokenimi modelimin kaçırmasını
# .. engellemek için log ile doğal bir scaleleme yapıyoruz. log alınca değerler belli bir aralığa sıkışıyor

# SONUÇ: Örneğin Bu 2 değeri çarpıyoruz : TF * IDF = 0,507 diyeceğiz

# Örnek hesaplama: yorum 100 kelimeden oluşsun "cow" kelimesi 3 kere geçsin ve 10 milyon dökümanda 1000 defa geçiyor olsun "cow" kelimesi
# Katsayı hesabı = 0.03 * 4 = 0.12
# TF-IDF data özelinde bu token önemli veya değil anlamında bir insight verir. (Yani örneğin:Olumlu-olumsuz şeklinde insight vermez)
# Yani TF-IDF özelinde katsayısı düşük olan tokenimiz de bizim için önemli bir keyword olabilir.

# Uygulamada yapacağımız aşamalar
# Data cleaning yapacağız
# Metni, sayısal forma dönüştüreceğiz
# Modele verip sonuç alacağız

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 50)
#!pip install nltk # Not: anaconda kullananların install etmesine gerek yok
# Nltk kütüphanesi bazı sürümlerde bu alttakilerin download edilmesini istiyor.
# Yani nltk yi import ettiğiniz halde hala hata alıyorsanız bunları download ediniz.
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

sample_text = "Oh man, this is pretty cool. We will do more such things. 2 ½ % ()"
# sayısal değerleri ve özel karakterleri nasıl temizlediğini görmek için bazı şeyler ekledik cümleye

# sentence tokenlere ayır ve küçük harflere dönüştür
sentence_token = sent_tokenize(sample_text.lower())
sentence_token
# Sent_tokenize: sentence tokenizer. Bunu çok kullanmayacağımızı söylemiştik Noktayı dikkate alarak ayırma işlemi yapıyor.
# SofiaH: sent_tokenize --> Sentence' lari bozmadan tokenize islemini yapar ve lower ile bütün harfleri kücük harfe cevirir

# word tokenlere ayır ve küçük harflere dönüştür
word_token = word_tokenize(sample_text.lower())
word_token
# word_tokenize: Boşlukları dikkate alarak ayırma işlemi yapıyor(NOT: Sadece string ifadelerde boşluk arar)
# SofiaH  word_tokenize --> Kelime kelime ayrim yaparak tokenize islemini yapar ve lower ile bütün harfleri kücük harfe cevirir.
# .. Noktalama isaretleri de birer token olarak kabul edilir

# Removing Punctuation and Numbers
# .isalnum() for number and object
tokens_without_punc = [w for w in word_token if w.isalpha()]
tokens_without_punc
# isalpha = Check if all the characters in the text are letters
# isalpha() : Çektiğim ifadenin string ise TRUE deyip(... list comprehension la kelimeleri alıyoruz)
# Bu aşamada Noktalama işaretleri, sayısal değerler ve özel karakterleri temizliyoruz
# String le birlikte sayısal değerlerinde kalmasını istiyorsanız .isalnum() kullanılabilir
# SofiaH: Tokenization isleminden sonraki ikinci asama olarak noktalama isaterlerinden ve sayilardan kurtulmamiz gerekiyor.
# SofiaH: ML ile hazirlanan modellerde classification veya sentimental analysis yapilabiliyor. Bu analizlerde de sayilar ve
# .. noktalama isaretlerini temizlemek gerekir.
# SofiaH: isalpha --> Tokenin object (str) ifade olup olmadigina bakar; object ise gecirir fakat noktalama isareti veya sayisal deger ise
# .. gecirmez. (Sayilarin da kalmasi isteniyor ise isalpha yerine .isalnum() yazilabilir.

# Removing Stopwords
# SofiaH: Cleaning islemini iki farkli sekilde yapacagiz : Classification islemi icin, sentimental analysis icin(olumlu veya olumsuz sonuç
# ..  bizim icin onemli ise).
# SofiaH: stop_words isimli degiskenin icine hangi dilin stopword' lerini kullanacaksak onu tanimladik
stop_words = stopwords.words("english")
stop_words

# SofiaH: Corpus' u (data) word tokenler haline getirmistik. Bu datadan stopword' leri cikaracagiz
tokens_without_punc

# if you don't make a sentiment analysis ,
token_without_sw = [t for t in tokens_without_punc if t not in stop_words]
# you can remove negative auxiliary verb
token_without_sw  # Output : ['oh', 'man', 'pretty', 'cool', 'things']
# stop_words lerde gez eğer cümlem(tokens_without_punc) içinde stop words(stop_words değişkeni) içinde değilse liste içerisinde tut, değilse ignore
# NOT: Eğer sentiment analizi yapmayacaksanız negatif yardımcı fiilleri çıkarabilirsiniz
# .. Eğer sentiment analizi yapacaksak bunları datada tutmamız lazım çünkü(no, not vs..) örneğin;
# It is a problem     (Olumsuz sonuç gelecek)
# It is not a problem (Olumlu sonuç gelecek)

# Data Normalization-Lemmatization
# SofiaH: NLP' de data normalization islemi Lemmatization veya Stemming ile yapilir. Lemmatization sözlükteki anlami korudugu icin daha cok
# .. tercih edilen bir yöntemdir.
WordNetLemmatizer().lemmatize("drives")
# Örnek: "driving" yazsaydık bunun farklı anlamı olduğunu bildiği için kökenine inmeyip olduğu gibi sonuç verirdi
# Örnek: "drove" yazarsak bunun "sürü" anlamı olduğunu da bildiği için olduğu gibi gelirdi
# Örnek: Children --> child olacak

lem = [WordNetLemmatizer().lemmatize(t) for t in token_without_sw]
lem  # Output : ['oh', 'man', 'pretty', 'cool', 'thing']
# SofiaH: Stopword' lerden temizlenmis corpus' un icindeki her bir tokeni list comprehension yöntemi ile köklerine indirmis olduk

# Data Normalization-Stemming
PorterStemmer().stem("children")  # Output : 'children'

stem = [PorterStemmer().stem(t) for t in token_without_sw]
stem  # Output: ['oh', 'man', 'pretti', 'cool', 'thing']
# Her bir tokenin köklerine iniyoruz
# Not: Çoğul ekinin de anlamı olmadığı için atmış

# Joining
# lem içerisindeki tokenleri al aralarında birer boşluk bırak ve birleştir
" ".join(lem)
# SofiaH: Liste icindeki bütun tokenleri join ile birlestirdik
# Her birini tek tek yapmak yerine bunu bir fonksiyona bağlayalım

# Cleaning Function - for classification (NOT for sentiment analysis)
def cleaning(data):
    # 1. Tokenize and lower
    text_tokens = word_tokenize(data.lower())
    # 2. Remove Puncs and numbers
    tokens_without_punc = [w for w in text_tokens if w.isalpha()]
    # 3. Removing Stopwords
    tokens_without_sw = [t for t in tokens_without_punc if t not in stop_words]
    # 4. lemma
    text_cleaned = [WordNetLemmatizer().lemmatize(t)
                    for t in tokens_without_sw]
    # joining
    return " ".join(text_cleaned)
# Classification yaparken bu fonksiyonu kullanabiliriz.
# SofiaH: Eger bir classification islemi yapacaksak, herhangi bir duygu analizi yapmayacaksak bu fonksiyonu kullanabiliriz


# Example: df["text"].apply(cleaning) # Output: 0    oh man pretty cool thing
pd.Series(sample_text).apply(cleaning)
# apply ı serilere uygulayabildiğimiz için seriye dönüştürdük.

# %% NLP-2
# Cleaning Function - for sentiment analysis
# SofiaH: Eger sentimental bir analiz yapacaksak olumlu-olumsuz yardımcı fiilerin text' in icinde kalmasi önemlidir.
sample_text = "Oh man, this is pretty cool. We will do more such things. don't aren't are not. no problem"

s = sample_text.replace("'", '')
word = word_tokenize(s)
word
# SofiaH: Asagida "Text' in icinde bir (') var ise bunun yerine hicbir sey atama diyerek bunu bir degiskene("s") atadik.
# SofiaH: Bu degiskeni word_tokenize icine vererek tokenlerine ayirdik. Ayraci kaldirdigimiz icin arent kelimesi stopword icindeki
# .. aren't kelimesi ile eslesmeyecek ve stopword isleminden sonra da bu kelimeler corpus icinde olmaya devam edecek

# SofiaH: Bazen aren't yerine are not da kullanilabilir. Bu ayri yazimlarin da stopword asamasinda temizlenmesini engellememiz gerekir.
# .. Bunun icin asagida bir fonksiyon tanimladik.
# SofiaH: Ilk olarak, yardimci fiillerdeki ayraclari kaldirdik.
# SofiaH: Ikinci olarak, bunlari word_tokenlerine ayirdik ve kücük harflere dönüstürdük.
# SofiaH: Ücüncü olarak, numaralardan ve noktalama isaretlerinden temizledik.
# SofiaH: Dördüncü olarak, stopword asamasinda bir for döngüsü kurarak 'not' ve 'no' sözcüklerini stopword' ler arasindan kaldirdik ki bu kelimeler
# .. stopword isleminden sonra da datamizda kalmaya devam etsin. Daha sonra list comprehension ile stopword' lerden temizleme islemini yaptik.
# SofiaH: Besinci olarak lemmatization islemi ile tokenlerin köklerine indik.
for i in ["not", "no"]:
    stop_words.remove(i)


def cleaning_fsa(data):
    # 1. removing upper brackets to keep negative auxiliary verbs in text
    text = data.replace("'", '')
    # 2. Tokenize
    text_tokens = word_tokenize(text.lower())
    # 3. Remove special characters numbers
    tokens_without_punc = [w for w in text_tokens if w.isalpha()]
    # 4. Removing Stopwords
    tokens_without_sw = [t for t in tokens_without_punc if t not in stop_words]
    # 5. lemma
    text_cleaned = [WordNetLemmatizer().lemmatize(t)  for t in tokens_without_sw]
    # joining
    return " ".join(text_cleaned)

stop_words

# SofiaH: sample_text' i seri haline getirmeden apply islemi uygulayamiyoruz. Text' i seri haline getirdikten sonra yukarida olusturdugumuz
# .. fonksiyonu apply ile ekledik. Böylece tek satirda cleaning islemi tamamlanmis oldu :
pd.Series(sample_text).apply(cleaning_fsa)

# CountVectorization and TF-IDF Vectorization
df = pd.read_csv("airline_tweets.csv")
df.head()
# SofiaH: Bir havayoluyla ilgili atilan tweet yorumlarindan olusan bir corpus var. Bu corpus üzerinden CountVectorization ve TF-IDF Vectorization
# .. islemlerinin mantiginin nasil isledigini görecegiz.

df = df[['airline_sentiment', 'text']]
df  # Sofia H: NLP datalarinin hepsi, text ve label olarak 2 feature' a düsürülür. Bu yüzden corpus' tan sadece bu iki sütunu aldik

df = df.iloc[:8, :]
# SofiaH: df' teki ilk 8 satiri ve tüm feature' lari aldik (Daha anlasilir olmasi icin) :
df

# SofiaH:  df' in bir kopyesini df2' ye atadik. (Bu sekilde yapmazsak hata veriyor) :
df2 = df.copy()

df2["text"] = df2["text"].apply(cleaning_fsa)
df2
# SofiaH: df2' deki text'e apply ile yukarida olusturdugumuz cleaning_fsa fonksiyonunu uyguladik
# .. ve boylece df icin claning islemini uygulamis olduk (Duygu analizi icin olusturdugumuz fonksiyon).
# NOT: SofiaH: !! Model kurarken cumle icindeki grammer yapisindan dolayi sira onemli fakat cumleler arasi sira onemli degil. _!!

# CountVectorization
# SofiaH: CountVectorizer ile text' leri sayisal hale dönüstürme islemi yapacagiz.
X = df2["text"]                   # Yorumlar
y = df2["airline_sentiment"]      # Target label

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=42)
# Datamizda toplam 8 cumle var, bunlari train ve test olarak yari yariya ayirdik

vectorizer = CountVectorizer()
X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)
# SofiaH: ML ve DL' deki scale isleminde yaptigimiz islemleri yapiyoruz; X_train'e fit_transform, X_test'e sadece transform islemi. (Data leakage'i
# .. engellemek icin)
# SofiaH: X_train'e fit uygulandiginda, X_train icindeki unique bütün tokenler tespit edilir; transform ile ise döküman icindeki her token sayilir.
# SofiaH: X_test'e transform islemi uygulandiginda, dökümandaki sayma islemlerini X_train' e göre yapar. Örnegin X_test' te 'car' kelimesi var
# .. fakat X_train' de bu kelime yoksa bu kelimeyi es gecer. Cünkü egittigimiz döküman icinde car kelimesi gecmiyor.
# SofiaH: Yani transform islemi, X_train' deki unique tokenlere göre yapilir.
# SofiaH: Bu yüzden X_train' i olabildigince buyuk tutmak gerekir ki tum tokenleri içersin.

# SofiaH: vectorizer' da egitilen unique token isimleri, feature isimleri olarak atandi :
vectorizer.get_feature_names()
# SofiaH: vectorizer.get_feature_names_out() --> Yeni versiyonlarda boyle.

# X_train' i array' e cevirdik ve her döküman icindeki tokenlerin teker teker sayildigini görmüs olduk
X_train_count.toarray()

df_count = pd.DataFrame(X_train_count.toarray(),
                        columns=vectorizer.get_feature_names())
# SofiaH: Array halindeki X_train datasini DafaFrame' e dönüstürdük. Columns isimleri olarak da get_feature_names' leri verdik.
df_count
# .. Her dökümanda her tokenin kac kere gectigini görüyoruz ::

X_train  # SofiaH: Yukaridaki DataFrame ile kiyaslamak icin asagida gercek X_train datasini yazdirdik, kelimelerin gercekte kacar kere gectigini
# .. kiyaslamis olduk

X_train[6]   # Train 0. indexteki cumle.
X_test[3]    # Test  0. indexteki cumle.
# SofiaH: vectorizer.vocabulary_ --> X_train' de gecen token sayilari.
vectorizer.vocabulary_

# TF-IDF
# sklearn TD-IDF https://towardsdatascience.com/how-sklearns-tf-idf-is-different-from-the-standard-tf-idf-275fa582e73d
tf_idf_vectorizer = TfidfVectorizer()
X_train_tf_idf = tf_idf_vectorizer.fit_transform(X_train)
X_test_tf_idf = tf_idf_vectorizer.transform(X_test)
"""
# SofiaH:
# TfidfVectorizer()' i bir degiskene atadik. Yine fit ve transform islemlerini yaptik.
# Yukaridaki CountVectorizer ile yaptigimiz fit_transform islemi ile buradaki farkli. 
# fit dedigimizde;
   # 1. Her satirda gecen unique tokenleri tespit eder (kac defa gectigine bakmaz), 
   # 2. Her cumlede gecen tokenleri sayar, bunlar her satirda var mi yok mu buna bakar (saymaz). 
# transform dedigimizde; 
   # 1. Unique tokenler kac satirda geciyor,
   # 2. Ilgili token her cumlede kac defa geciyor?
# ..  islemlerini yapar. Bu islemlerden sonra buldugu sayilari TF-IDF formullerinde yerlerine koyarak tokenleri sayisal veriler haline dönüstürür.

# NOT : X_train icin yaptigi transform islemini X_test icin de yapar fakat X_test' te gecen bir token X_train' de gecmiyorsa o tokeni görmezden
# .. gelir. Boyle bir durumda IDF hesabi yapilirken deger yerine konuldugunda log(0/0)=sonsuz olacaktir. Bunun onune gecmek icin IDF, 
# .. degerlere 1 ekler. log((0+1)/(0+1)). Bu sekilde degerlerin NaN cikmasinin önüne gecer.
"""

# SofiaH: tf_idf_vectorizer.get_feature_names_out()
tf_idf_vectorizer.get_feature_names()

X_train_tf_idf.toarray()

df_tfidf = pd.DataFrame(X_train_tf_idf.toarray(),
                        columns=tf_idf_vectorizer.get_feature_names())
# SofiaH: DataFrame'e cevirip unique token isimlerini verdik ve yeni feature' lar olustu :
df_tfidf

X_train[6]  # Output: 'virginamerica yes nearly every time fly vx ear worm go away'
# SofiaH: X_train' in 6. degerini yine kiyaslama icin aldik (Sifirinci index' teki deger). Yukarda ilk indexe bakacak olursak virginamerica
# .. tokeni neredeyse her satirda gectigi icin stopword gibi kabul edilmis ve agirligi azaltilmis. Bir token her satirda gecerse önemsizlesir.
# .. Bu tokenlere classification yapilamaz. (CountVectorizer, virginamerica kelimesini onemsizlestirmemisti fakat TF-IDF onemsizlestirdi)

# SofiaH: Datanin 2. indexinde var olan kelimelerden en dusuk agirliga sahip olan tokeni, virginamerica
df_tfidf.loc[2].sort_values(ascending=False)

pd.DataFrame(X_test_tf_idf.toarray(),
             columns=tf_idf_vectorizer.get_feature_names())
# SofiaH: X_testi DataFrame donusturduk (X_train' deki feature' lara gore)

X_test
X_test[3]
# SofiaH: X_test' in ilk cumlesi 3. cumle. Bu cumlede gecen aggressive kelimesinin X_test feature' lari arasinda olmadigini goruyoruz.
# .. Demek ki bu kelime X_train' de yokmus ve gözardi edilmis. Bu durumda tahmin asamasinda modelin tahmin yapmasi zorlasir.
# .. Bu yuzden X_train' i olabildigince fazla datayla egitmek onemlidir.

###### NLP Application with ML
# Classification of Tweets Data
# SofiaH: Bu calismada ML modellerinin NLP ile nasil kullanildigina dair ornekler yaptik. (NLP' de DL modelleri ML modellerine gore daha cok tercih edilir)
# The data Source: https://www.kaggle.com/crowdflower/twitter-airline-sentiment?select=Tweets.csv

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (10, 6)
pd.set_option('display.max_columns', 50)

df = pd.read_csv("airline_tweets.csv")
df.head()
# SofiaH: Havayolu sirketlerine ait atilan tweet' lerden olusan bir data. Datada yolcularin isimleri, yaptiklari yorumlar,
#  havayolu sirketlerinin adi, tarih gibi bilgiler mevcut. Target label' da neutral, positive, negative olmak uzere
# .. 3 farkli class var. Butun NLP modellerinde sadece target ve text onemlidir, sadece bu ikisi ile islem yapacagiz.
# SofiaH: ÖNEMLİ NOT: !!! NLP' de missing value' lar doldurulmaz, silinir. !!!

# !pip install matplotlib = 3.4
# Matplotlib versiyonu 3.4 ten küçükse,
# for p in ax.containers:
#     ax.bar_label(p)
# .. kodu çalışmayacaktır

ax = sns.countplot(data=df, x="airline", hue="airline_sentiment")
for p in ax.containers:
    ax.bar_label(p)
# SofiaH: Hangi havayoluna kac sikayet gelmis grafigini cizdirdik. Cok yogun sikayet alan havayolu sirketleri var
# Her havayoluna yapılan ürünlerin kaçı positive-negative-neutral
# Her bar da birden fazla label varsa for döngüsünü kullanabiliriz(Kullanarak 3 bar-3 bar şeklinde gösterdi)
# united, Us Airways, American firmaları genelde(3 te 2 si) olumsuz yorum almışlar. Müşterilerin bizden kaçmaması için bir şeyler yapmamız gerekli

ax = sns.countplot(data=df, x="negativereason")
ax.bar_label(ax.containers[0])  # tek label. for döngüsüne gerek yok
plt.xticks(rotation=90)      # X eksenindeki isimleri 90 derece açıyla yazsın
# SofiaH: negativereason feature ina gore bir grafik cizdirdik. Musterilerin cogu, musteri hizmetlerinden, gec ucuslardan vb. sikayetci diyebiliriz

ax = sns.countplot(data=df, x="airline_sentiment")
ax.bar_label(ax.containers[0])
# SofiaH: En fazla yorum negatif olarak yapilmis (Dengesiz bir dataseti)
# Hedef label ım negative yorumlar. Benim için önemli bu. Müşterinin kaçmaması için bunlara yoğunlaşacağız

# negative    9178 neutral     3099 positive    2363
df["airline_sentiment"].value_counts()

# Cleaning Data
df2 = df.copy()
df2["text"].head()  # Temizlenmemiş datamız

# Cleaning Data
# Cleaning islemi icin regex kutuphanesini kullanacagiz.
s = "http\\:www.mynet.com #lateflight @airlines"
s = re.sub("http\S+", "", s).strip()
s
# Önceki otomatik yaptığımız fonksiyona(cleaning) ilaveten bazı işlemler yapmalıyız
# tag ler, mention lar ve link lerin bize hiç bir katkısı yok.
# re.sub --> Regexin bir fonksiyonudur. Asagida bu fonksiyon ile "http ile basla, bosluga kadar butun karakterleri temizle." demis
# .. olduk. com' dan sonra gelen bosluga kadar her seyi temizledi, sonraki kisimlar kaldi. Cumlenin basinda ve sonundaki bosluklari kaldirmak
# .. icin de strip() kullandik :
# "http\S+", "" : http den sonra, boşluk "hariç" en az bir kere veya daha fazla karakter varsa hepsini al ve onların(\:www.mynet.com) yerine
# .. boşluk getir

s = re.sub("#\S+", "", s)
s
# "#" ile baslayan tum ifadeleri boşluk hariç en az bir karakter varsa bosluga kadar temizle (strip yazmayinca basta bosluk kaldi)

s = re.sub("@\S+", "", s)
s.strip()

# "@" ile baslayan tum ifadeleri boşluk hariç en az bir karakter varsa bosluga kadar temizle
# Yukarida temizledigimiz kelimeler her satirda oldugu icin modelde gürültüye sebep olur ve temizlenmeleri gerekir, egitime de bir katkilari yoktur

# Sentimental analiz yapacagimiz icin not ve no kelimeleri analiz icin gerekli. for dongusu ile bu kelimeleri stopwords' ler
# .. arasindan cikardik

"""negative_auxiliary_verbs = ["no", 'not', "n't", 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
                            "doesn't", "don't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 
                            'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',"needn't", 'shan', "shan't", 'shouldn',
                            "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', ,"won't", 'wouldn', "wouldn't"]"""

stop_words = stopwords.words('english')
for i in ["not", "no"]:
    stop_words.remove(i)
# Sentimental analiz yapacagimiz icin not ve no kelimeleri analiz icin gerekli. for dongusu ile bu kelimeleri stopwordsler arasindan cikardik

# Tum cleaning islemlerini yapabilmek icin asagida cleaning isimli bir fonksiyon tanimladik.
# Yukarida tek tek yaptigimiz kaldirma islemlerini asagida ilk 3 satirda yaptik.


def cleaning(data):
    import re
    # 1. Removing URLS
    data = re.sub('http\S+', '', data).strip()
    data = re.sub('www\S+', '', data).strip()
    # 2. Removing Tags
    data = re.sub('#\S+', '', data).strip()
    # 3. Removing Mentions
    data = re.sub('@\S+', '', data).strip()
    # 4. Removing upper brackets to keep negative auxiliary verbs in text
    data = data.replace("'", "")
    # 5. Tokenize
    text_tokens = word_tokenize(data.lower())
    # 6. Remove Puncs and number
    tokens_without_punc = [w for w in text_tokens if w.isalpha()]
    # 7. Removing Stopwords
    tokens_without_sw = [t for t in tokens_without_punc if t not in stop_words]
    # 8. lemma
    text_cleaned = [WordNetLemmatizer().lemmatize(t)
                    for t in tokens_without_sw]
    # joining
    return " ".join(text_cleaned)
# 4.islemde doesn't gibi kelimeleri ust ayractan kurtardik ki stopword uygulandiginda bu kelimeler corpus' ta bulunmaya devam etsin.
# 5.islemde tum kelimeleri kucuk harflere donusturerek tokenize ettik.
# 6.islemde noktalama isaretleri ve sayilari cikardik.
# 7.islemde corpusu stopword' lerden temizledik.
# 8.islemde tokenlerin köklerine indik.
# Son olarak join ile tüm tokenleri birlestirdik.


cleaning_text = df2["text"].apply(cleaning)
cleaning_text.head()
# Create ettigimiz fonksiyon icine df' teki 'text' sütununu apply fonksiyonu ile verdik ve cleaning islemlerini tamamladik

##### Features and Label
df2 = df2[["airline_sentiment", "text"]]
df2.head()
# Modelde kullanacagimiz iki sütunu aldik. text' in temizlenmemis halini aldik cunku asagida vectorizer fonksiyonu icinde bir parametre
# .. ile cleaning islemini yapacağız

# Train Test Split
X = df2["text"]
y = df2["airline_sentiment"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=101)
# X ve y' nin %20' sini test olarak ayirdik. Dengesiz bir dataseti oldugu icin stratify=y kullandik
# Johnson H: Skorların 0.2 ile iyi olduğunu gördüğüm için, test_size ı daha çok küçültmedik
# .. Eğer sonuçlarınız kötü gelirse test size ınızı küçülterek deneyin

# Vectorization
# CountVectorizer islemine tabi tutulan corpus ile Logistic Regression, SVM, KNN, RF, AdaBoost modellerini kuracagiz, daha sonra
# .. TF-IDF islemine tabi tutulan corpus ile yine tum modelleri kuracagiz ve en iyi skor aldigimiz modeli sececegiz.
# ngram_range=(1,2), max_features= 1500
vectorizer = CountVectorizer(preprocessor=cleaning, min_df=3)
X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)
# Vectorizer Parametreleri :
# preprocessor --> Yukarida create ettigimiz fonksiyon olan cleaning' i bu parametreye tanimlarsak, cleaning islemini yapar.
# min_df=3 --> Corpus' ta 3' ten daha az(2 veya 1) gecen tokenleri egitime dahil etme. (Egitime katkisı olmaz bunların)
# max_features=1500 --> Corpusta en fazla kullanilan ilk 1500 feature ı dikkate al. (Bunun yerine min_df kullanilmasi tavsiye edilir;
# .. max_features kullanmak egitime engel olabilir, risklidir.). Burada da ilk 2000 önemlidir belki. Bunun sınırını belirlemek zor.
# .. Johnson H: Kesinlikle tavsiye etmiyorum
# max_df=1000 --> corpusta 1000 den fazla geçen tokenleri ignore et. (Kullanilmasi tavsiye edilmez.
# .. Farkinda olmadan egitime katkisi olacak tokenler cikarilabilir. Bu sınırı belirlemek zor.) Johnson H: Kesinlikle tavsiye etmiyorum.
# .. Bunun yerine min_df kullanilmasi tavsiye edilir;
# ngram_range(1,2) --> Cumledeki kelimeleri bir tek tek alir bir de ilk 2 kelimeyi alir, bir kaydirir, sonraki 2 kelimeyi alir sona kadar
# .. bu islemi yaparak modeldeki kaliplari ogrenmeye calisir. Fakat bu sekilde de feature sayisi cok artacagi icin egitim islemi cok uzar.
# .. Genel olarak (1,2) veya (1,3) olarak kullanilir, daha fazlasi tavsiye edilmez. (1,3) secildiginde (1,1), (1,2), (1,3)' u de yapar ve
# .. feature sayisi cok artar. Guclu bir makine varsa bu parametre tercih edilebilir.

# Temizleme işlemini CountVectorizer içinde yapıyoruz. Temizledikten sonra datayı sayısal vektörlere dönüştürür
# datamızda ngram(1,1) dir. Her tokeni 1 er 1 er ayırır. --> Ahmet Tv'yi çok beğendi(4 feature)
#           ngram(1,2) olsaydı                           --> "Ahmet Tv'yi" bir token , "Tv'yi çok" bir token, "çok beğendi" bir token oluyor
# .. bu da kalıpları(sıfat tamlaması, isim tamlaması ..vs) daha iyi anlamasını sağlıyor. Skorlar kötü ise ngram ı (1,2), ya da (1,3) yaparak deneyin
# ngram(1,3) --> secildiginde (1,1), (1,2), (1,3)' u de yapar ve feature sayisi cok artar(Aradaki sayıları da artar)
# Johnson hoca: (1,3) ü geçmenizi tavsiye etmiyorum. ngram(1,1) çoğu zaman yeterli oldu.
# class chat soru: bu featurelar düğerlerinin üzerine mi ilave olacak yani hem 1-1 hem 1-2 featureları mı gösterecek
# Johnson H: evet hocam 2'li tokenlerden oluşan yeni featurlar ilave olur df'mimize
# class chat soru: Bu(ngram olayı) sadece sentiment analysis için mi geçerli?
# Johnson H: yok hocam clasification ve sentimant analyisi farketmiyor
# class chat: Hocam burada pipeline kullanabiliyor muyuz?
# Johnson H: Kullanabilirsiniz

X_train_count.toarray()

pd.DataFrame(X_train_count.toarray(),
             columns=vectorizer.get_feature_names_out())
# CountVectorizer ve fit-transform islemlerinden sonra X_train' i array' e cevirdik ve token isimlerini alarak DataFrame' e donusturduk.
# .. 3126 tane feature' imiz var. Burada Feature Importance islemi yapamayiz fakat PCA yontemi kullanmak mantikli olur, o sekilde feature
# .. sayisi 100, 200 gibi bilesen sayisina dusurulebilir :
# NOT: Eğer üstte min_df =3 ü kaldırırsak 7757 feature a çıktı. Bazı datalarda 100000 lere çıkabilir
# NOT: Ram iniz düşükse bu aşamalarda hata alabilirsiniz bazı datalarda

# Model Comparisons - Vectorization


def eval(model, X_train, X_test):
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    print(confusion_matrix(y_test, y_pred))
    print("Test_Set")
    print(classification_report(y_test, y_pred))
    print("Train_Set")
    print(classification_report(y_train, y_pred_train))

##### Naive Bayes
# https://medium.com/analytics-vidhya/na%C3%AFve-bayes-algorithm-5bf31e9032a2
# P(A|B) --> A olayi gerceklestiginde B olayinin gerceklesme ihtimali nedir? Buradaki A olayi "don't like" olsun. don't ve like
# .. tokenleri birlikte kullanildiginda yorumun negatif olma olasiligi nedir durumuna Naive Bayes bakar. (P(don't like|negative)).
# .. Olasiliklar uzerinden calistigi icin NLP' de cok guzel sonuclar verir. Bu yuzden NLP ile kullanilmasi tavsiye edilir.
# Navie Bayes' te kullanilan 3 farkli algoritma var. Bunlardan en cok kullanilan iki tanesi : MultinomialNB, BernoulliNB
# MultinomialNB --> Daha cok multiclass datalarda tercih edilir.
# BernoulliNB --> Daha cok binary datalarda tercih edilir. ANCAK;Dokumaninda her ikisinin de denenip hangisi iyi sonuc veriyorsa
# .. onun secilmesi onerilir)
nb = MultinomialNB(alpha=10)
nb.fit(X_train_count, y_train)
# Bu datada multiclass ve MultinomialNB ile daha iyi sonuc verdigi icin bununla devam edecegiz.
# alpha --> Mesela ilk satirda 'able' tokeni hic gecmemis (0). O zaman olasilik hesabi yaparken P(0|positive)=0 cikacak.
# ..  alpha, bu tur tokenlere bir regularization islemi yapar. 'able' kelimesi gectigi zaman da corpus icindeki kullanim sıklıklarına
# .. gore dusuk de olsa mutlaka bir olasilik döndürür. Bu tokene dusuk bir agirlik verilmis olur fakat bu token cumlede geciyorsa
# ..  positive veya negative olma durumu ogrenilmis olur. Bu sayede Naive Bayes modellerin overfite gitmesi engellenir.
# .. alpha degeri ne kadar buyurse o kadar yuksek derecede regularization islemi uygulanir (Ridge ve Lasso' daki gibi) (Default=1)
# Logistic Regression' da ci degeri, SVM' deki gamma degeri kuculdukce uygulanan regularization islemi artiyordu, burda ise alpha degeri ne
# .. kadar buyurse uygulanan regularization islemi o kadar artar.
# !!! Eger bir overfit durumuyla karsilasilirsa yapilacak ilk islem, alpha degerini buyutmektir !!!
"""        olumlu        olumsuz
müthiş       10             1
mükemmel     8              1      
iyi          7              1
tv          15             12
iyi          8             13
telefon      2             10     
kötü         2             8 
kargo        4             3   
Corpusumda 15 olumlu, 10 olumsuz yorum var

P(müthiş tv kötü kargo | olumlu)  = 10/66 * 15/66 * 2/66 * 4/66 * 15/25 = 3,79 * 10^-5
P(müthiş tv kötü kargo | olumsuz) = 1/49 * 12/49 * 8/49 * 4/49 * 10/25 = 3,79 * 10^-5
"""

# Naive Bayes teoremi daha cok NLP modelleri ile kullanilir. Bayes teoremine göre çalışır
# P(I don't like | olumlu) Yorumumda "I", "don't" ve "like" tokenleri geçerken olumlu olma olasılığı
# P(I don't like | olumsuz) Yorumumda "I", "don't" ve "like"  tokenleri geçerken olumsuz olma olasılığı
# P(I don't like | nötr) Yorumumda "I", "don't" ve "like"  tokenleri geçerken nötr olma olasılığı
# .. bunlara bakar hangi olasılık büyükse o sınıfa atar tokeni. "0.5" gibi bir threshold yoktur
# Örnek: P(kötü | olumsuz) : kötü tokeni geçerken olumsuz olma olasılığı # P(olumsuz) : Olumsuz yorum olma olasılığı
# .. "Müşhiş" in bir yorumu olumsuz yapma olasılığı 1/49
# NOT: alpha nın rolü(alpha=10 için): Eğer datam "harika tv kötü kargo" olsaydı --> harika kelimesi hiç geçmediği için olumlu
# .. olma olasılığını "0"(0/66=0) a çeker) direk olasılığı 0 a çekmesin diye alpha değerini dikkate alır pay a
# .. alpha değerini ekler paydaya da "15(olumlu yorum)(yada  2.yol: unique token sayısı=8) * alpha" değerini ekler
# ..  --> (0+10) / 66 + (15*10)  yani alpha bir # NOT: 2. yol ile yapsaydık    0+10 / 66 + (8*10) şeklinde yapacaktık
# .. smoothness işlemi yapar yani olasılığın 0 a gitmesini engeller. Bunu o sıradaki HER TOKENE(!!!) uyguluyoruz. Yani, hesaplama;
# P(harika tv kötü kargo | olumlu)  = 10/216 * 25/216 * 12/66 * 14/66 * 25/175 = ... a dönüyor (Bunu tabi olumsuz için de yapacağız aynı mantık ile)
# Sonuç olarak alpha bunu sağlar ayrıca alpha ile oynayarak bu overfitting i giderebiliriz

print("NB MODEL")
eval(nb, X_train_count, X_test_count)
# alpha=10 degeri ile train ve test datalarindaki negatif skorlarin birbirlerine yaklastigini gorduk, bu yuzden bu alpha degerini
# .. sectik (Negatif skorlar ile ilgileniyoruz). Train set ve Test set skorlari birbirine yakin, overfitting durumu yok
# Test:  negative       0.71      0.98      0.83      1835
# Train: negative       0.72      0.98      0.83      7343
# Johnson H: overfitting kontrolünü her zaman cross validation üzerinden yapın

scoring = {'accuracy': make_scorer(accuracy_score),
           'precision-neg': make_scorer(precision_score, average=None, labels=["negative"]),
           'recall-neg': make_scorer(recall_score, average=None, labels=["negative"]),
           'f1-neg': make_scorer(f1_score, average=None, labels=["negative"])}

model = MultinomialNB(alpha=10) # !!! model üzerinde hem fit, hem prediction yaptıysak data leakage i engellemek için modelimizi cv den önce yeniden kuruyorduk
# .. çünkü fit ve prediction işlemleri yapılan model üzerinde hem train, hem test bilgilerini içerdiği için data leakage e sebep olabilir
scores = cross_validate(model, X_train_count, y_train,
                        scoring=scoring, cv=10, return_train_score=True)
df_scores = pd.DataFrame(scores, index=range(1, 11))
df_scores.mean()[2:]
# NOT: labels içerisine birden fazla sınıf yazmazsak "average=None" ın hiç bir anlamı yok. average="macro", average="micro" ,average="weighted"
# .. hepsi aynı sonuç verecektir. Çünkü çünkü labels içerisinde "1" tane label var o da "negative"
# test: validation(1 parça) , train deki train(9 parça)
# Overfitting kontrolünü yaptık
# Hem de skorların tutarlılığını kontrol ettik.
# Johnson H: Her zaman bu şekilde kontrol edin

viz = PrecisionRecallCurve(
    MultinomialNB(alpha=10),   # model
    classes=nb.classes_,      # targettaki class' lar
    per_class=True,           # class isimlerini goster
    cmap="Set1"               # renklendirme
)
viz.fit(X_train_count, y_train)     # egitim yap
viz.score(X_test_count, y_test)    # skorlari al
viz.show()                        # gorsellestir
# Dengesiz bir datasetimiz oldugu icin precision_recall ile modelin skorlarina baktik. Negative class icin modelin genel performansi %92
# PR for class negative(area=0.92):  Benim labellarım negative labelları %92 oranında başarılı bir şekilde ayrıştırıyor

y_pred = nb.predict(X_test_count)
nb_count_rec_neg = recall_score(y_test, y_pred, labels=["negative"], average=None)
nb_count_f1_neg = f1_score(y_test, y_pred, labels=["negative"], average=None)
nb_AP_neg = viz.score_["negative"] # Modelin genel performansını gösteren skor(negative için Precision recall curve skoru)
# Modelleri bir tabloda karsilastiracagimiz icin; negative label icin recall_score, f1 score ve gorseldeki skoru(viz.score ile) alip
# .. birer degiskene atadik

##### Logistic Regression
log = LogisticRegression(C=0.02, max_iter=1000)  # class_weight='balanced'
log.fit(X_train_count, y_train)
# C değerimizin büyük olmasını istemiyorduk. Overfitting varsa C değerini küçültüyoruz
# C değeri küçülürse uyguladığı regresyon kuvveti artar
# max_iter=1000 : Log Reg arkada Gradient Descent kullanır. Hatanın min e inmesi için iterasyonu kullanır. iterasyon yetersizse bu arttıralır
# .. uyarı vermeyene kadar arttırabilirsiniz
# class_weight='balanced' .. negative in eğitim için sayısı yeterli, ilgilendiğim skorlarda negatif skorlarda iyi o yüzden class_weight='balanced' yapmadık

print("LOG MODEL")
eval(log, X_train_count, X_test_count)
# negative       0.76      0.96      0.85      1835
# negative       0.77      0.97      0.86      7343
# Johnson H: Müşteriye bu skorlar verilir mi verilmez mi? Bunlar nasıl yorumlanır? Overfitting var mı? En optimal skorlar nedir?
# .. En iyi skorları almaya çalışmamalıyız. Önemli olan bu soruların cevaplarını doğru verebiliyor muyuz? Yorum yapabiliyor muyuz?
# .. Sonuç olarak: Metriklerin yorumlanması en önemli kısımdır. Mesele kodları çalıştırmak değildir. Müşteriye verilen
# .. skorlar optimal değilse yazdığınız kodların, aldığınız skorların hiç bir anlamı yok

model = LogisticRegression(C=0.02, max_iter=1000)
scores = cross_validate(model, X_train_count, y_train,
                        scoring=scoring, cv=10, return_train_score=True)
df_scores = pd.DataFrame(scores, index=range(1, 11))
df_scores.mean()[2:]
# Not(ÖNEMLI): "df_scores" ile  : validation ve train setindeki bütün skorları siz yine de bir gözden geçirin(Burada ortalama aldık sadece(df_scores.mean()[2:]))
# Ancak skorlar yakın olmayabilir cv yaparken bu sefer müşteriye ben genelleme yapıyorum dersiniz
# .. sonra müşteri farklı skorlar alabilir. Bunun kontrolünü mutlaka yapın(Alttaki 2 resme bkz.)
# Yani Skorları yakın olan modeller arasında seçim yapmakta zorlanılıyorsa
# .. Validation ve train seti arasındaki farkların en düşük olduğu modeli almamız daha doğru
# NOT: Skorlar yakınken model genelleme yapabiliyor deriz(Cv de)(Precision: 0.76,0.8,0.77,0.75,0.75, yada recall için 0.97,0.96,0.94,0.93,0.94
# NOT: Skorlar yakın değilken model genelleme yapamıyor deriz(Cv de)(Yani ) Precision: 0.7,0.8,1,1,0.6, recall için 0.82,0.71,0.94,0.7,0.69

viz = PrecisionRecallCurve(
    LogisticRegression(C=0.02, max_iter=1000),
    classes=log.classes_,
    per_class=True,
    cmap="Set1"
)
viz.fit(X_train_count, y_train)
viz.score(X_test_count, y_test)
viz.show()
# Modelim negative labelları/class ları diğer labellardan %92 oranında başarılı bir şekilde ayırıyor
# Modelleri karşılaştırma yaparken öncelikle bakacağımız precision,recall, f1 score lardır.
# .. Sonra precision-recall curve(modelin genel performansı) e bakıyoruz
# PrecisionRecallCurve' de yukaridakine benzer bir skor elde ettik

y_pred = log.predict(X_test_count)
log_count_rec_neg = recall_score(
    y_test, y_pred, labels=["negative"], average=None)
log_count_f1_neg = f1_score(y_test, y_pred, labels=["negative"], average=None)
log_AP_neg = viz.score_["negative"]  # Modelin genel performansını gösteren skor(negative için Precision recall curve skoru)
# Karsilastirma icin skorlari degiskenlere atadik

log = LogisticRegression(C=0.02, max_iter=1000, class_weight='balanced')
log.fit(X_train_count, y_train)
# Modeldeki negative skorlarla ilgileniyoruz fakat yine de class_weight='balanced' dedik ve neutral ve positive skorlar da yukseldi.
# .. neutral ve positive yorumlar da onemli ise class_weight='balanced' mutlaka kullanilmali, cunku dengesiz bir datasetimiz var
# class_weight='balanced' yaptığımız zaman inbalanced olan sınıfların skorları iyileşirken fazla olan(negative) sınıfın
# .. skorları düşecektir(GENELDE)

print("LOG MODEL BALANCED")
eval(log, X_train_count, X_test_count)

# %% NLP-3
# SVM
# Linear i tercih ettik default yerine. Default biraz maliyetli idi
svc = LinearSVC(C=0.01)
svc.fit(X_train_count, y_train)
# C=1 default degeri ile model overfite gittigi icin bu degeri asama asama kuculterek en iyi skoru aldigimiz 0.01' i sectik

print("SVC MODEL")
eval(svc, X_train_count, X_test_count)
# Bazı data scientist ler başarıyı sadece recall üzerinden değerlendirir ama mutlaka precision ı da değerlendirmek gerekir
# Recall=1, precision=0,1 olursa. recall=100 hastanın hepsini bulmuş, precision 1000 tanesinin 100 hepsine hasta(1) demiş. Bu bir başarı değildir
# .. Müşteri o 1000 kişi için tekrar testler yapmak zorunda kalacak(Müşteriye maliyet)
# Evet recall u yüksek elde etmeye çalışıyoruz ama precision değeri de önemli.
# .. Precision daki Yüzde 1 lik değişmeler bile müşteri için maliyet açısından değerli olabilir
# precision benim için ne zaman önemli
# Modelin hasta dediği hasta dediği 100 de 100 ü de hasta(Precision = 1) olsun derse müşteri precision ı kullanırız
# .. Yani müşteri modele hastaları sokunca gerçekten hasta olanları bulur
# recall un mantığı ise yine bütün kanser hastalarını bulayım ama min tahminle bulayım(maliyeti daraltayım)
# Modellerde skorları((test ve train de)) aynı ise(Precision, recall vs), hangisi(hangi model) daha iyi genelleyebiliyor(hangi modelin train,test skorları en yakınsa) o modeli kullanırız

model = LinearSVC(C=0.01)
scores = cross_validate(model, X_train_count, y_train, scoring=scoring, cv=10)
df_scores = pd.DataFrame(scores, index=range(1, 11))
df_scores.mean()[2:]
# Tek seferlik skorlar ile CrossValidate sonucu alinan skorlar dengeli

viz = PrecisionRecallCurve(
    LinearSVC(C=0.01),
    classes=svc.classes_,
    per_class=True,
    cmap="Set1"
)
viz.fit(X_train_count, y_train)
viz.score(X_test_count, y_test) # !!!  viz.score_ dan gelen sonıç ortalamadır
viz.show()
# Modelin genel performansi onceki modeller ile cok yakin
# !!! NOT: Önceliğimiz precision, recall, f1 score daha sonra modelin genel performansına bakıyoruz. Model negatif skorları iyi ayırıyor olabilir
# .. ama önce precision, recall, f1 score a bakıp sonra diğerlerine bakılmalı.
# 2 model için her şey aynı ise , hangisi daha iyi genelleyebiliyorsa(cv skorları hangisinin daha yakınsa(validation ve train) o model seçilir
# Ancak eğer model precision, recall, f1 score kötü ama modelin genel performansı iyi ise o zaman o uğraşmaya değer bir modeldir yani,
# .. precision, recall, f1 score arttırılabilire işaret eder. Bizim için önemli olan hedef label ın(negative skorları(precision, recall, f1 score)
# .. eğer positive ve neutral skorları kötü ise bu benim için önemli değildir. Bütün labellar önemli olsaydı, macro skorlarına bakardık

y_pred = svc.predict(X_test_count)
svc_count_rec_neg = recall_score(y_test, y_pred, labels=["negative"], average=None)
svc_count_f1_neg = f1_score(y_test, y_pred, labels=["negative"], average=None)
svc_AP_neg = viz.score_["negative"]  # !!!  viz.score_ dan gelen sonıç ortalamadır
# Karsilastirma icin skorlari degiskenlere atadik

##### KNN
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_count, y_train)
# KNN icin Elbow metodu ile egitim cok uzun surecegi icin ve skorlar da cok kotu ciktigi icin bir kaç n_neighbors degeri deneyerek
# .. n_neighbors=7' de karar kildik ve skorlarimizi aldik

print("KNN MODEL")
eval(knn, X_train_count, X_test_count)

model = KNeighborsClassifier(n_neighbors=7)
scores = cross_validate(model, X_train_count, y_train, scoring=scoring, cv=10)
df_scores = pd.DataFrame(scores, index=range(1, 11))
df_scores.mean()[2:]
# CrossValidate sonucu alinan skorlar da oldukca kotu

viz = PrecisionRecallCurve(
    KNeighborsClassifier(n_neighbors=7),
    classes=knn.classes_,
    per_class=True,
    cmap="Set1"
)
viz.fit(X_train_count, y_train)
viz.score(X_test_count, y_test)
viz.show()
# Buradaki skor 0,83 ancak bunu direk atalım demeyebiliriz. Yani overfit i giderirsek bu skorlar
# .. yükselebilir o yüzden bizim önceliğimiz buradaki 0,83 değil önce precision,recall, f1 skorlarımızdır
# .. oradaki overfit i engellersek buradaki skor da yükselecektir
# KNN modelin genel performansi onceki modellerden oldukca dusuk cikti

y_pred = knn.predict(X_test_count)
knn_count_rec_neg = recall_score(y_test, y_pred, labels=["negative"], average=None)
knn_count_f1_neg = f1_score(y_test, y_pred, labels=["negative"], average=None)
knn_AP_neg = viz.score_["negative"]
# Karsilastirma icin aldigimiz skorlari degiskenlere atadik

##### Random Forest
# class_weight="balanced"
rf = RandomForestClassifier(100, max_depth=40, random_state=42, n_jobs=-1)
rf.fit(X_train_count, y_train)
# GridSearch islemi yapmadik. RF modellerde ilk oynamamiz gereken parametre olan max_depth ile oynadik. 100, 200 gibi degerler verdigimizde
# .. modeli overfitten kurtaramadik; degerler kuculdukce negative skorlar icin overfitin engellendigini gorduk. En iyi skoru 40 degerinde aldik

print("RF MODEL")
eval(rf, X_train_count, X_test_count)
# Ağaç yöntemlerinde Overfit i engellemek için önceliğimiz max_depth dir. 2. önemli parametre ağaç sayısıdır(100 yazan yer/n_estimators)

model = RandomForestClassifier(100, max_depth=40, random_state=42, n_jobs=-1)
scores = cross_validate(model, X_train_count, y_train, scoring=scoring, cv=10)
df_scores = pd.DataFrame(scores, index=range(1, 11))
df_scores.mean()[2:]
# Tek seferlik skorlarimiz ile CrossValidate sonucu aldigimiz skorlar tutarli

viz = PrecisionRecallCurve(
    RandomForestClassifier(100, max_depth=40, random_state=42, n_jobs=-1),
    classes=rf.classes_,
    per_class=True,
    cmap="Set1"
)
viz.fit(X_train_count, y_train)
viz.score(X_test_count, y_test)
viz.show()
# Negative class icin modelin genel performansi yuksek

y_pred = rf.predict(X_test_count)
rf_count_rec_neg = recall_score(
    y_test, y_pred, labels=["negative"], average=None)
rf_count_f1_neg = f1_score(y_test, y_pred, labels=["negative"], average=None)
rf_AP_neg = viz.score_["negative"]
# Karsilastirma icin skorlari degiskenlere atadik

rf = RandomForestClassifier(
    100, max_depth=40, random_state=42, n_jobs=-1, class_weight="balanced")
rf.fit(X_train_count, y_train)
# RF modele bir de class_weight='balanced' ile baktik. positive' de train ve test setleri arasinda yuksek fark var.
# .. Positive skorlara bakacak olsaydik, yukaridaki parametreler ile denemeler yapıp modeli overfit durumundan kurtarmamiz gerekirdi

print("RF MODEL BALANCED")
eval(rf, X_train_count, X_test_count)
# Buradaki overfit i gidermek için max_depth le oynayabiliriz ama bizim için önemli olan "balanced" sonucu değil.

# Ada Boost
ada = AdaBoostClassifier(n_estimators=500, random_state=42)
ada.fit(X_train_count, y_train)

print("Ada MODEL")
eval(ada, X_train_count, X_test_count)
# Ada Boost yerine XGBoost da tercih edilebilirdi. n_estimators= 500 ile train ve test skorlarinin birbirine yaklastigini gordugumuz icin
# .. bu degeri sectik.
# precision ve recall skorları dengeli. Müşteri bu skorların dengeli olmasını isterse Bu modeli kullanabilirsiniz
# !!! recall değerini arttır dememize rağmen skorlar dengeli gelmiş

model = AdaBoostClassifier(n_estimators=500, random_state=42)
scores = cross_validate(model, X_train_count, y_train, scoring=scoring, cv=10)
df_scores = pd.DataFrame(scores, index=range(1, 11))
df_scores.mean()[2:]
# Tek seferlik skorlar ile CrossValidate sonucu alinan skorlar birbirleriyle tutarli

viz = PrecisionRecallCurve(
    AdaBoostClassifier(n_estimators=500, random_state=42),
    classes=ada.classes_, # NOT: classes=nb.classes_ # classes_ yazmazsak label isimleri(negative, positive, neutral) yerine 0,1,2 görünür çıktıda
    per_class=True,
    cmap="Set1"
)
viz.fit(X_train_count, y_train)
viz.score(X_test_count, y_test)
viz.show()
# Negative class icin modelin genel performansi yuksek. 

y_pred = ada.predict(X_test_count)
ada_count_rec_neg = recall_score(
    y_test, y_pred, labels=["negative"], average=None)
ada_count_f1_neg = f1_score(y_test, y_pred, labels=["negative"], average=None)
ada_AP_neg = viz.score_["negative"]
# Karsilastirma icin skorlari degiskenlere atadik
# Yukaridaki islemler ile CountVectorizer kismini tamamlamis olduk, TF-IDF kismina gececegiz


"""
# ÖNEMLİ HATIRLATMA: Johnson H: Overfitting kontrolünü her zaman Cross validation da aldığımız skorların birbirine yakın olup olmamasına 
# .. bakarak yapıyoruz.(Yani Alttaki kod a göre)

model = AdaBoostClassifier(n_estimators= 500, random_state = 42)
scores = cross_validate(model, X_train_count, y_train, scoring = scoring, cv = 10,return_train_score=True)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]

# Bu kodun çıktısındaki ;
# .. test accuracy/f1/recall= "validation" skorlarının ortalama sonucu(10 değerin ortalaması), 
# .. train test accuracy/f1/recall= "train içindeki train" skorlarının ortalama sonucu(10 değerin ortalaması)

# Yani "eval(ada, X_train_count, X_test_count)" kodunda aldığımız test ve train sonuçlarının birbirine
# .. yakın olup olmamasına bakarak DEĞİL!!

Genellemeyi doğru yapabilen cv ye örnek(Bu resmin çıktı halini "df_scores" yazdırarak alabiliriz. "df_scores.mean()[2:]" yazdığımız için ortalama
.. alınmış sonuçları görüyoruz) .. df_scores daki skorlarında yani ortalama alınmadan olan skorların da birbirine yakın olmasına dikkat edin

test_accuracy          0.714482
train_accuracy         0.734138
test_precision-neg     0.704816
train_precision-neg    0.717308
test_recall-neg        0.978891
train_recall-neg       0.984293
test_f1-neg            0.819534
train_f1-neg           0.829855
# Skorlar birbirine yakın overfitting yok --> (0.714482- 0.734138) , (0.704816 0.717308),  (0.978891 0.984293), (0.819534 0.829855)
# Müşteriye vereceğimiz skorlar da alttakiler
"eval(ada, X_train_count, X_test_count) --> buradaki test skorları

Test_Set
              precision    recall  f1-score   support
    negative       0.71      0.98      0.83      1835
     neutral       0.75      0.21      0.33       620
    positive       0.84      0.44      0.57       473

    accuracy                           0.73      2928
   macro avg       0.77      0.54      0.57      2928
weighted avg       0.74      0.73      0.68      2928
# En sonda da cv deki skorlarla buradaki(üstteki çıktıdaki) negative skorları yakın mı ona bakmalıyız 
"""

##### TF-IDF
# Yukarida CountVectorizer ile text' i sayisal verilere donusturup tum modeller icin skorlar aldik. Simdi ise sayisal verilere donusturme
# .. islemini TF-IDF ile yapacagiz ve tum modeller icin yine skorlar alacagiz.
tf_idf_vectorizer = TfidfVectorizer(preprocessor=cleaning, min_df=3)
X_train_tf_idf = tf_idf_vectorizer.fit_transform(X_train)
X_test_tf_idf = tf_idf_vectorizer.transform(X_test)
# preprocessor=cleaning diyerek create ettigimiz fonksiyon ile cleaning islemini yapmis olduk. (min_df=3 kullanmak idealdir.)
# !!!min_df=3 . Benim corpusumda  1 veya 2 YORUMDA geçen tokenleri ignore et demek(Corpusumda 1 veya 2 kere geçen tokenleri ignore et demek DEĞİL)
# DIKKAT bu şu demek DEĞİL: Benim corpusumda 1 veya 2 kere geçen tokenleri ignore et..
# class chat soru: Hocam datadan 1 veya 2 gibi ya da belli bir değerin altında frekansa sahip olan kelimeleri silmeli miyiz?
# Johnson H: O da yöntemlerden birtanesi ama bizim min_df ile yaptığımız zaten datada 1 veya 2 defa geçenleride siliyor. 
# .. Yani min_df daha geniş kapsamlı bir işlem yapıyor
"""
Johnson H notları
# TF-IDF fit'in yaptığı işlem train setindeki unique bütün tokenleri tespit eder (tüm dönüşümler train setindeki unique
# tokenlere göre yapılır)
# TF-IDF transformun yaptığı işlem;
# Her unique tokenin her yorumda kaçar defa geçtiğini tespit eder (Hem traim hemde test seti için ayrı ayrı) TF pay.
# Her yorumun(dokument, row) kaç tokenden oluştuğunu tespit eder (Hem traim hemde test seti için ayrı ayrı) TF payda.
# Her unique tokenin kaç satırda (document) geçtiğini tespit eder (Hem traim hemde test seti için ayrı ayrı) DF pay.
# datanın toplam kaç satırdan oluştuğunu tespit eder (Hem traim hemde test seti için ayrı ayrı)(DF payda) ve formulde yerlerine
# koyup hesaplamasını yapar.
"""

X_train_tf_idf.toarray()

pd.DataFrame(X_train_tf_idf.toarray(), columns=tf_idf_vectorizer.get_feature_names_out())
# get_feature_names_out(): fit işleminin uygulandığı train setindeki unique tokenleri isimleri

##### Model Comparisons TF-IDF
##### Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_tf_idf, y_train)
# MultinomialNB() icindeki alpha parametresi ile oynamadik cunku default deger olan 1 ile negative class icin en iyi skoru verdi

print("NB MODEL")
eval(nb, X_train_tf_idf, X_test_tf_idf)
# Modelde overfit durumu yok

model = MultinomialNB()
scores = cross_validate(model, X_train_tf_idf, y_train, scoring=scoring, cv=10)
df_scores = pd.DataFrame(scores, index=range(1, 11))
df_scores.mean()[2:]
# CrossValidate skorlari tek seferlik skorlar ile tutarli

viz = PrecisionRecallCurve(
    MultinomialNB(),
    classes=nb.classes_,
    per_class=True,
    cmap="Set1"
)
viz.fit(X_train_tf_idf, y_train)
viz.score(X_test_tf_idf, y_test)
viz.show()
# Modelin genel performansı yüksek

y_pred = nb.predict(X_test_tf_idf)
nb_tfidf_rec_neg = recall_score(
    y_test, y_pred, labels=["negative"], average=None)
nb_tfidf_f1_neg = f1_score(y_test, y_pred, labels=["negative"], average=None)
nb_tfidf_AP_neg = viz.score_["negative"]

# Logistic Regression
log = LogisticRegression(C=0.4, max_iter=1000)
log.fit(X_train_tf_idf, y_train)
# C = 0.4 degeri ile model overfitten kurtuldugu icin bu degeri sectik.

print("LOG MODEL")
eval(log, X_train_tf_idf, X_test_tf_idf)

model = LogisticRegression(C=0.4, max_iter=1000)
scores = cross_validate(model, X_train_tf_idf, y_train, scoring=scoring, cv=10)
df_scores = pd.DataFrame(scores, index=range(1, 11))
df_scores.mean()[2:]
# Tek seferlik skorlar ile CrossValidate sonucu alinan skorlar tutarli

viz = PrecisionRecallCurve(
    LogisticRegression(C=0.4, max_iter=1000),
    classes=log.classes_,
    per_class=True,
    cmap="Set1"
)
viz.fit(X_train_tf_idf, y_train)
viz.score(X_test_tf_idf, y_test)
viz.show()
# Negative class icin modelin performansi yuksek

y_pred = log.predict(X_test_tf_idf)
log_tfidf_rec_neg = recall_score(
    y_test, y_pred, labels=["negative"], average=None)
log_tfidf_f1_neg = f1_score(y_test, y_pred, labels=["negative"], average=None)
log_tfidf_AP_neg = viz.score_["negative"]
# Karsilastirma icin skorlari degiskenlere atadik

log = LogisticRegression(C=0.4, max_iter=1000, class_weight="balanced")
log.fit(X_train_tf_idf, y_train)

print("LOG MODEL BALANCED")
eval(log, X_train_tf_idf, X_test_tf_idf)
# class_weight='balanced' ile skorlarimiza tekrar baktik. Positive class' in train ve test skorlari arasinda fark var, bu farki azaltmak
# .. icin parametreler ile oynamak gerekir fakat bizim icin onemli class Negative class' i oldugu icin oynamayacagiz

# SVM
svc = LinearSVC(C=0.1)
svc.fit(X_train_tf_idf, y_train)
# SVM modelde C=0.1 ile negative class'taki overfitting' i giderebildik.

print("SVC MODEL")
eval(svc, X_train_tf_idf, X_test_tf_idf)

model = LinearSVC(C=0.1)
scores = cross_validate(model, X_train_tf_idf, y_train, scoring=scoring, cv=10)
df_scores = pd.DataFrame(scores, index=range(1, 11))
df_scores.mean()[2:]
# Tek seferlik skorlar ile CrossValidate sonucu alinan skorlar birbirine yakın

viz = PrecisionRecallCurve(
    LinearSVC(C=0.1),
    classes=svc.classes_,
    per_class=True,
    cmap="Set1"
)
viz.fit(X_train_tf_idf, y_train)
viz.score(X_test_tf_idf, y_test)
viz.show()
# Negative class' a ait genel performans yuksek

y_pred = svc.predict(X_test_tf_idf)
svc_tfidf_rec_neg = recall_score(
    y_test, y_pred, labels=["negative"], average=None)
svc_tfidf_f1_neg = f1_score(y_test, y_pred, labels=["negative"], average=None)
svc_tfidf_AP_neg = viz.score_["negative"]
# Karsilastirma yapmak uzere skorlari degiskenlere atadik

# KNN
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_tf_idf, y_train)

print("KNN MODEL")
eval(knn, X_train_tf_idf, X_test_tf_idf)
# KNN modeli skorlari onceki gibi cok kotu oldugu icin parametreler ile oynamanin da bir anlami yok

model = KNeighborsClassifier(n_neighbors=7)
scores = cross_validate(model, X_train_tf_idf, y_train, scoring=scoring, cv=10)
df_scores = pd.DataFrame(scores, index=range(1, 11))
df_scores.mean()[2:]
# Modelin genel performansi cok dusuk

viz = PrecisionRecallCurve(
    KNeighborsClassifier(n_neighbors=7),
    classes=knn.classes_,
    per_class=True,
    cmap="Set1"
)
viz.fit(X_train_tf_idf, y_train)
viz.score(X_test_tf_idf, y_test)
viz.show()

y_pred = knn.predict(X_test_tf_idf)
knn_tfidf_rec_neg = recall_score(
    y_test, y_pred, labels=["negative"], average=None)
knn_tfidf_f1_neg = f1_score(y_test, y_pred, labels=["negative"], average=None)
knn_tfidf_AP_neg = viz.score_["negative"]
# Karsilastirma yapmak uzere skorlari degiskenlere atadik

##### RandomForest
rf = RandomForestClassifier(100, max_depth=40, random_state=42, n_jobs=-1)
rf.fit(X_train_tf_idf, y_train)
# Parametreler ile oynayarak en iyi skoru aldigimiz parametreleri sectik

print("RF MODEL")
eval(rf, X_train_tf_idf, X_test_tf_idf)

model = RandomForestClassifier(100, max_depth=40, random_state=42, n_jobs=-1)
scores = cross_validate(model, X_train_tf_idf, y_train, scoring=scoring, cv=10)
df_scores = pd.DataFrame(scores, index=range(1, 11))
df_scores.mean()[2:]
# Tek seferlik skorlar ile CrossValidate sonucu alinan skorlar birbirine yakin :

viz = PrecisionRecallCurve(
    RandomForestClassifier(100, max_depth=40, random_state=42, n_jobs=-1),
    classes=rf.classes_,
    per_class=True,
    cmap="Set1"
)
viz.fit(X_train_tf_idf, y_train)
viz.score(X_test_tf_idf, y_test)
viz.show()
# Negative class' a ait genel performans yuksek

y_pred = rf.predict(X_test_tf_idf)
rf_tfidf_rec_neg = recall_score(y_test, y_pred, labels=["negative"], average=None)
rf_tfidf_f1_neg = f1_score(y_test, y_pred, labels=["negative"], average=None)
rf_tfidf_AP_neg = viz.score_["negative"]

rf = RandomForestClassifier(
    100, max_depth=15, random_state=42, n_jobs=-1, class_weight="balanced")
rf.fit(X_train_tf_idf, y_train)

print("RF MODEL BALANCED")
eval(rf, X_train_tf_idf, X_test_tf_idf)
# class_weight="balanced" secildiginde positive skorlar birbirine biraz daha yaklasti

##### Ada Boost
ada = AdaBoostClassifier(n_estimators=500, random_state=42)
ada.fit(X_train_tf_idf, y_train)

print("Ada MODEL")
eval(ada, X_train_tf_idf, X_test_tf_idf)
# Negative class' lar icin en dengeli skoru AdaBoost modelde aldik, precision ve recall skorlari birbirlerine cok yakin. Musteri dengeli bir
# .. skor istediginde bu model sunulabilir

model = AdaBoostClassifier(n_estimators=500, random_state=42)
scores = cross_validate(model, X_train_tf_idf, y_train, scoring=scoring, cv=10)
df_scores = pd.DataFrame(scores, index=range(1, 11))
df_scores.mean()[2:]
# Tek seferlik skorlar ile CrossValidate sonucu alinan skorlar birbirine yakin

viz = PrecisionRecallCurve(
    AdaBoostClassifier(n_estimators=500, random_state=42),
    classes=ada.classes_,
    per_class=True,
    cmap="Set1"
)
viz.fit(X_train_tf_idf, y_train)
viz.score(X_test_tf_idf, y_test)
viz.show()
# precision ve recall degerleri diger modellere gore daha dusuk ciktigi icin genel performans da %90 cikmis

y_pred = ada.predict(X_test_tf_idf)
ada_tfidf_rec_neg = recall_score(y_test, y_pred, labels=["negative"], average=None)
ada_tfidf_f1_neg = f1_score(y_test, y_pred, labels=["negative"], average=None)
ada_tfidf_AP_neg = viz.score_["negative"]

# Compare Scoring
compare = pd.DataFrame({"Model": ["NaiveBayes_count", "LogReg_count", "SVM_count", "KNN_count", "Random Forest_count",
                                  "AdaBoost_count", "NaiveBayes_tfidf", "LogReg_tfidf", "SVM_tfidf", "KNN_tfidf",
                                  "Random Forest_tfidf", "AdaBoost_tfidf"],

                        "F1_Score_Negative": [nb_count_f1_neg[0], log_count_f1_neg[0], svc_count_f1_neg[0], knn_count_f1_neg[0],
                                              rf_count_f1_neg[0], ada_count_f1_neg[0], nb_tfidf_f1_neg[0], log_tfidf_f1_neg[0],
                                              svc_tfidf_f1_neg[0], knn_tfidf_f1_neg[0], rf_tfidf_f1_neg[0], ada_tfidf_f1_neg[0]],

                        "Recall_Score_Negative": [nb_count_rec_neg[0], log_count_rec_neg[0], svc_count_rec_neg[0],
                                                  knn_count_rec_neg[0], rf_count_rec_neg[0], ada_count_rec_neg[0],
                                                  nb_tfidf_rec_neg[0], log_tfidf_rec_neg[0], svc_tfidf_rec_neg[0],
                                                  knn_tfidf_rec_neg[0], rf_tfidf_rec_neg[0], ada_tfidf_rec_neg[0]],

                        "Precision_Recall_Score_Negative": [nb_AP_neg, log_AP_neg, svc_AP_neg, knn_AP_neg, rf_AP_neg,
                                                            ada_AP_neg, nb_tfidf_AP_neg, log_tfidf_AP_neg, svc_tfidf_AP_neg,
                                                            knn_tfidf_AP_neg, rf_tfidf_AP_neg, ada_tfidf_AP_neg]})
# compare # Bu bir df, bu df teki skorları biz büyükten küçüğe doğru sıralayıp görselleştireceğiz

# !!! NOT: nb_count_f1_neg --> Bu skor array içerisinde bunu array içinde çıkarmak için nb_count_f1_neg[0] şeklinde yazıyoruz
# !!! NOT: nb_AP_neg --> Bu array içerisinde gelmediği için nb_AP_neg[0] şeklinde yazmadık
# Tum modellerden elde ettigimiz skorlari kiyaslamak amaciyla bir fonksiyon create ettik.
# compare degiskeni icine olusturdugumuz tum model isimlerini tanimladik. Bununla birlikte yukarida negative class' i icin aldigimiz tum f1 score,
# ..  recall score ve precision recal score' lari da tanimladik. (recall ve f1 score, precision hakkinda da inside sagladigi icin onu yazmadik)
# Tanimladigimiz fonksiyon ile 3 ayri grafik elde ettik; tum modeller icin ilk grafik recall, ikinci grafik f1 score, ucuncu grafik ise precision
# .. recall skorlarini temsil ediyor.
# recall grafiginde en yuksek skoru RF modeli verdi.
# f1 score grafiginde yuksek skorlari SVM modelleri verdi fakat LogReg_tfidf ile de arasinda cok fazla bir fark yok. LogReg_tfidf' in recall
# .. skoru da %95. RF model recall grafiginde en yuksek skoru vermisti fakat f1 grafiginde skorlari dusuk. Hem recall hem f1 skorunun yuksek
# .. olmasini, bununla birlikte modelin genel performansinin da yuksek olmasini istiyoruz.
# Hem hizli calistigi icin hem de f1 skoru yuksek oldugu icin LogReg_tfidf secmek daha mantikli.
# LogReg_tfidf' in genel performansina baktigimizda %92, en yuksek skor ile arasinda cok fazla bir fark yok. Model olarak LogReg_tfidf' i secmeye
# .. karar verdik
# NOT: Aşağıdaki görselleri elde etmek için dict oluşturduk

def labels(ax):
    for p in ax.patches:
        width = p.get_width()                        # get bar length
        ax.text(width,                               # set the text at 1 unit right of the bar
                p.get_y() + p.get_height() / 2,      # get Y coordinate + X coordinate / 2
                # set variable to display, 2 decimals
                '{:1.3f}'.format(width),
                ha='left',                         # horizontal alignment
                va='center')                       # vertical alignment


plt.figure(figsize=(15, 30))
plt.subplot(311) # 3 satır 1 sütun , 1. görselde gösterilecek olan
compare = compare.sort_values(by="Recall_Score_Negative", ascending=False)
ax = sns.barplot(x="Recall_Score_Negative", y="Model",
                 data=compare, palette="Blues_d")
labels(ax)

plt.subplot(312) # 3 satır 1 sütun , 2. görselde gösterilecek olan
compare = compare.sort_values(by="F1_Score_Negative", ascending=False)
ax = sns.barplot(x="F1_Score_Negative", y="Model",
                 data=compare, palette="Blues_d")
labels(ax)

plt.subplot(313) # 3 satır 1 sütun , 3. görselde gösterilecek olan
compare = compare.sort_values(
    by="Precision_Recall_Score_Negative", ascending=False)
ax = sns.barplot(x="Precision_Recall_Score_Negative",
                 y="Model", data=compare, palette="Blues_d")
labels(ax)
plt.show()

# 1. görselde Recall skorlarına göre büyükten küçüğe doğru sırala ve negatif skorları bar plot ile görselleştir
# 2. görselde f1 skorlarına göre büyükten küçüğe doğru sırala ve negatif skorları bar plot ile görselleştir
# 3. görselde Precision-recal_score a(Model genel performansına) göre büyükten küçüğe doğru sırala ve negatif skorları bar plot ile görselleştir
# Precision a neden bakmadık? Bir skorun recall u düşük f1 düşükse zaten precision düşük olacaktır
# .. Yani 2 sinden(precision, recall) dan biri düşükse f1 de düşeceği için burada precision a bakmaya gerek duymadık
# .. Siz isterseniz precision ı da ekleyip görselleştirebilirsiniz
# Yorumlar: Recallar yakın ilk görselde ama biz precision ların da yakın olmasına bakıyoruz
# Naive bayes recall da yüksek ama f1 de(yani aslında precisionda da) düşmüş
# Örneğin bir modelde  recall 0.98, preci: 0,7 . diğer modelde recall 0.96, preci:0,82 ise
# .. buradaki recall 2 puan düşük olanı tercih edebiliriz(precision da yakın olduğu için)

##### PREDICTION FOR PIPELINE FOR NEW TWEETS
pipe = Pipeline([('tfidf', TfidfVectorizer(preprocessor=cleaning, min_df=3)), ('log', LogisticRegression(C=0.4, max_iter=1000))])
# Kullanmaya karar verdigimiz modelimiz ile prediction yapacagiz. 
# Modeli tf-idf kullanarak textlerden temizle ve sayısal değerlere dönüştür sonra modeli uygula
# Pipeline ile fit_transform ve fit_prediction islemlerini yapabiliyorduk.Ilk yazdigimiz fit_transform, ikincisi ise fit_prediction islemlerini yapar.
# !!! NOT: Burada scale ihtiyacımız yok olsaydı onu da yazardık
# En iyi islemin TF-IDF olduguna karar vermistik, ilk olarak bunu tanimladik, ikinci kisma Logistic Regression' da en iyi sonucu aldigimiz
# .. parametreleri tanimladi
# !!! NOT: sklearn ün pipeline ı fit, transform ve predict işlemlerine izin verir

X.head()
pipe.fit(X, y) # Tüm data kullanıldı eğitim için
# Pipe icine X ve y' nin temizlenMEMIŞ halini veriyoruz, cunku TfidfVectorizer icine tanimladigimiz cleaning_fsa fonksiyonu ile bu işlemi yapacak.
# !!! NOT: Modelden bir prediction alabilmek icin sample' i mutlaka series' e donusturmek gerekir. !!!
# !!! NOT: Prediction icin verilen sample'da noktalama isareti veya ozel karakterler olsa bile pipe bu karakterleri temizler.

##### PREDICTION
tweet = "it was not the worst flight i have ever been."
tweet = pd.Series(tweet)  # !!! NOT: Eğitimi hangi formatta yaptıysak prediction da aynı formatta olmalı o yüzden seriye çevirdik
pipe.predict(tweet)  # Output : array(['negative'], dtype=object)
# Prediction icin verilen sample' da noktalama isareti veya ozel karakterler olsa bile pipe bu karakterleri temizleyecektir, ayrica
# .. temizlememize gerek yok.
# Modelimiz tweet' in negative bir yorum oldugunu bildi

tweet = "I don't like flight"
tweet = pd.Series(tweet)
pipe.predict(tweet)
# Modelimiz tweet' in negative bir yorum oldugunu bildi

tweet = "I like flight"
tweet = pd.Series(tweet)
pipe.predict(tweet)
# Modelimiz tweet' in positive bir yorum oldugunu bildi

tweet = "don't enjoy flight  at all"
tweet = pd.Series(tweet)
pipe.predict(tweet)
# Modelimiz tweet' in negative bir yorum oldugunu bildi

tweet = "I don't think I'll ever use American Airlines any more"
tweet = pd.Series(tweet)
pipe.predict(tweet)
# Modelimiz tweet' in negative bir yorum oldugunu bildi

tweet = "it is amazing flight"
tweet = pd.Series(tweet)
pipe.predict(tweet)
# Modelimiz  tweet' in positive yorum oldugunu bilemedi. Bir sonraki adimda bunun nedenini arastiracagiz

tweet = "I don't think I'll ever use American Airlines anymore"
tweet = pd.Series(tweet)
pipe.predict(tweet)
# Modelimiz  tweet' in negative yorum oldugunu bilemedi. Bir sonraki adimda bunun nedenini arastiracagiz

tweet = "ok flight"
tweet = pd.Series(tweet)
pipe.predict(tweet)

# Collect Words and Counting words
tweets = cleaning_text
tweets
# cleaning_text, yukarida CountVectorizer ile degil manuel olarak temizledigimiz text idi. Bir sonraki kodda bu yorumlarin hepsini
# .. join ile birlestirdik ve tweets degiskenine atadik

for i in enumerate(tweets):
    print(i) # Hem indexi hem de yorumu döndürecek


counter = 0
for i, j in enumerate(tweets):
    if "dont like" in j and y[i] == "negative":   # "don't like" ifadesi yorum içinde(j de) geçiyor mu ve yorum olumsuz mu
        counter += 1                              # üstteki 2 şartı sağlıyor ise counter ı 1 arttır. 
print(counter)                   # Sonundada gelen sayıda şunu tespit etmeye çalışıyoruz aslında--> Bu durum benim datamda kaç defa geçiyor/oluyor
# 14 defa geçtiği için datamda modelim bunu öğrenmiş ve üstte doğru tahmin yaptı

counter = 0
for i, j in enumerate(tweets):
    if "like" in j and y[i] == "positive":
        counter += 1
print(counter)
# Like yorumu geçen kaç tane pozitif yorumu var --> 66 .. E iyi doğru bilmesi lazım. AMA alta bakalım

counter = 0
for i, j in enumerate(tweets):
    if "like" in j and y[i] == "negative":
        counter += 1
print(counter)
# Like tokeninin geçtiği kaç tane olumsuz yorum var --> 304
# Model: like kelimesi olumlu ve olumsuz yorumlarda geçiyor ama olumsuz olma olasılığı daha yüksek
# O yüzden model yukardaki tahminleri yanlış yaptı
# Countvektozier ve TF-IDF de tokenlerin birbirleriyle olan anlamsal ilişkileri yakalayamıyor
# .. bunu wordembedding dersinde bakacağız bu konuya detaylı
# "like" bir keyword olmaktan çıktı. Çünkü hem olumlularda hem de olumsuzlarda sıklıkla geçiyor. Yani bir nevi stopword olmuş
# "dont like" bir keyword olabilir

counter = 0
for i, j in enumerate(tweets):
    if "isnt amazing" in j and y[i] == "negative":
        counter += 1
print(counter)
# output: 0 ... Datada "isnt" ile "amazing" hiç bir araya gelmemiş. Yani amazing e olumlu diyip amazing görünce olumlu deyip onu olumluya atıyor

# Yani sonuç olarak sonuçlarımız iyi ama DATAMIZ kötü. Müşteri geldiği zaman ona senin datan kötü diyebiliriz
# Sonuçlar kötü ise: 1. train datanız küçüktür 2.train datanızdaki unique tokenlerden kaynaklı olarak bazı tokenlerin predictionında sıkıntı olması
# .. 3.bazı tokenlerin bir arada kullanılmama durumu

##### WORDCLOUD - Repetition of Words
# Bu asamada corpus' ta siklikla kullanilmis olan tokenleri gorsellestirecegiz. Bunun icin wordcloud kutuphanesini import edecegiz.
# Collect Words
tweets = cleaning_text
tweets

all_words = " ".join(tweets)  # Bütün yorumları tek bir cümle gibi birleştirdik


all_words[:100] # Ilk 100 karakterine bakarak yorumlarin birlestigini teyit ettik

# Create Word Cloud
# !pip install wordcloud
# conda install -c conda-forge wordcloud
worldcloud = WordCloud(background_color="white", max_words=250)
# background_color : Arka plan rengini ayarlar.
# max_words = 250 : En fazla kullanilan 250 kelimeyi goster. (Baska sayilar verilebilir)

# generate ile en fazla kullanilan 250 kelimeye ait gorselin yapisi arka planda olusturulur.
worldcloud.generate(all_words)
# worldcloud.generate(all_words) : tokenleri gez generate ile en fazla kullanilan 250 kelimeye say

plt.figure(figsize=(13, 13))
# interpolation : renklendirme
plt.imshow(worldcloud, interpolation="bilinear",)
# Cerceve olsun mu?
plt.axis("off")
plt.show()
# Gorsellestirme icin matplotlib altyapisini kullandik. imshow icine olusturdugumuz wordcloud' i verince corpus icinde daha cok gecen
# .. ifadeler daha buyuk harfli olarak gorselestirilmis oldu

# Not: WordCloud'un içerisinde mask parametresi bulunuyor buraya background için istediğimiz resmi-şablonu koyabiliyoruz

##################################
# WORD EMBEDDING YONTEMI
# Tanım1: Word Embeddings, Kelimeler arasındaki anlamsal ilişkileri gösteren sayısal vektörlerdir
# Tanım2: Benzer anlama sahip kelimelerin benzer bir temsile sahip olmasına izin veren bir kelime temsili türüdür.
"""
model.wv['ankara']                                    model.wv['istanbul'] 
output: array([0.281,-0.400, 0.462, .....            output: array([0.32,-0.38, ....., ..... 
               -0.54, .................                ................................
               .........................               ...............................
               ........................])                 .............................)]
-- 0.281, -0.400 : Bu değerlerin her birine feature representation olarak geçer bir feature ı temsil ettikleri için.
-- Word Embedding, tokenlerin birbirlerine olan benzerliğine buradaki bu feature representation üzerindeki sayısal değerlerin birbirlerine 
-- .. yakınlığına göre karar veriyor.
-- Örneğin 2 tane tokenimiz olsun Ankara ve Istanbul tokenlerinin benzerliği yukardaki ilk 2 feature representation üzerinden yapılıyor diyelim
(NOT: Hepsini kullanmıyor 1-2 feature represantation ı kullanıyor)
-- Yani 1. feature representation üzerinden değerler: (0.281, 0.32) ve 2. feature representation üzerinden değerler:(-0.400, -0.38)
--  bu değerler birbirlerine çok yakın olduklarından modelim ankara ve istanbul için birbilerine çok yüksek anlamsal ilişki atayacaktır
-- Yani word embedding bu anlamsal ilişkileri sözlükteki anlamlarına göre değil, tokenlerin birbirleri ile kullanım sıklıklarına göre yakalıyor
"""
# Iki kelimenin birbirine anlamsal olarak yakın olduğunu bu kelime vektörlerinin cosinus similarity sine bakarak karar veririz
# cosinus similarity 1 e yaklaştıkça anlamsal olarak yakınlaşırlar 0 a yaklaştıkça anlamsal olarak yakın değiller demektir
# Elimizdeki wordlerin sayısal vektöre dönüştürülmesi konusunda genelde Word_embedding kullanılıyor(Diğer yöntemlerin handicapları var)
# SofiaH: CountVektorizer ve TF-IDF tokenlerin kendi aralarındaki anlamsal ilişkiyi yakalayamaz. Mesela
# .. güzel-çirkin arasındaki anlamsal ilişkiyi yakalayamaz
# SofiaH: Word Embedding tokenlerin kendi aralarındaki anlamsal ilişkiyi yakalar. Bu yüzden tercih edilir
# SofiaH: İki kelimenin veya cümlenin birbirine ne kadar yakın anlamda olduğu Cosinus similarity fonksiyonu ile bakılabilir
# SofiaH: Cosinus similarity değeri 1'e ne kadar yakınsa bu kelimeler anlamsal olarak birbirlerine o kadar yakın demektir

# Eğer 2 tokenin birlikte sık kullanıldıklarını görürse bunu yakalıyor word_embbeding
# Yani mesela deep ve learning 2 farklı kelime olsa bile bu 2 sini birlikte çok görürse buna anlamsal ilişki yüklüyor
# .. ve bu yapıyı öğreniyor.anlamsal ilişkiyi yakalıyor derken tokenlerin sıklıkla birbirleri ile kullanımlarını kastediyoruz

# Corpus um olsun ve altı tane cümle olsun:
# öğretmen öğrenciye okulda not verdi
# öğretmen talebeye not verdi
# öğretmen öğrenciye okulda ödev verdi
# eğitmen talebeye üniversitede not verdi
# eğitmen talebeye üniversitede proje yaptı
# eğitmen talebeye üniversitede ödev verdi
# burada geçen unique tokenleri bulalım : öğretmen, öğrenciye, okulda, "not verme"(Bunu birlikte alalım anlamsal ilişkiyi kaybetmemek adına),
# .. "proje yapmak", "ödev vermek", eğitmen, talebeye, üniversitede
# 9 tane unique token var.
# Bunların arasındaki feature representation ne olabilir. (Yani bu 9 tane tokeni ortak bir şekilde temsil eden)
# .. Eğitim olabilir, öğretim olabilir # NOT: Biz bu repesentation ların ne olduğunu göremiyoruz
# Eğitim den öğretmenin aldığı feature representation --> 0.95
# Eğitim den eğitmenin aldığı feature representation  --> 0.91
# Öğretim in öğretmenin aldığı feature representation --> 0.9
# Öğretim in eğitmenin aldığı feature representation  --> 0.88
# Şimdi burada word embedding diyor ki 0.95 ile 0.91 ya da 0.9 ile 0.88 birbirine yakın olduğu için
# öğretmen ve eğitmen arasındaki feature representation(1-2 tane olabilir) arasında anlamsal ilişkiler kuruyor
# .. yani anlamsal olarak birbirlerine yakın olduğun anlıyor(öğretmen ve eğitmenin)
# Bu anlamsal ilişkileri cosinus similarity ile belirliyoruz.
# class chat soru: Feature sayısı da otomatik belirleniyor öyle mi Hocam? Word embedding de
# Johnson H: Genelde 50, 100 ve 300 dür. Manuel olarak belirleriz. Ancak hazır modellerin word embeddinglerini değiştirme imkanı yoktur

# Vektörün boyutunun ne olacağına biz karar veriyoruz. Piyasada genelde 50,100,300 boyutlu vektörler tercih edilir
"""
Ders 3 , dakika 2:35:00
# datamdan 6 tane tokeni alalım

Gender: Man   = -1   Demekki gender ile %100 ilişkili bu iki kelime ama birbirlerinin tam zıttı
         Woman =  1
# Bir kişiye kadın veya erkek diyebilmem için "gender" feature ına ihtiyacı var. O yüzden 1 ve -1 demiş

        King   = -0.95  Model, King-Queen'in anlamsal olarak birbirine yakın olduğunu fakat gender özelliği üzerinden de birbirlerinin zıttı
        Queen  = 0.97   .. olduğunu anlamış
# FEATURE REPRESENTATION açıklaması: King ve queen arasındaki anlamsal ilişkiyi gender feature representation ı üzerinden çok güzel bir şekilde kurmuş!!!

        Apple  = 0.00  Model, bunların gender ile bir ilgisi olmadığını anlamış
        Orange = 0.01
# King ve Queen üzerindeki ilişkiyi gender, royal, age i kullanarak belirledik(3 tanesini kullandık)

Royal: Man   = 0.01    Model, bunların royal ile bir ilgisi olmadığını anlamış
       Woman = 0.02

       King   = 0.93  Model, King-Queen'in kraliyetle direk ikişili olduğunu anlamış bu yüzden
        Queen  = 0.95  .. zıt işaretler yok

.......

Age:   King = 0.7     King-Queen olabilmek için belli bir yaşa gelmek gerekir. Bu yüzden age ile
       Queen = 0.69   .. güçlü bir ilişki var. Erkek veya bayan olmanızla yaşın bir etkisi yok

.....

Food:  Apple = 0.95, 
        Orange = 0.97 

# SONUÇ: 
# man ve woman üzerindeki ilişkiyi gender ı kullanarak belirledik(1 tanesini kullandık). Yani man ve woman birbirlerinin yerine kullanılmış vs
# King ve Queen üzerindeki ilişkiyi gender, royal, age i kullanarak belirledik(3 tanesini kullandık)
# man ve woman birbirlerinin yerine kullanılabilir, ya da sıklıkla birlikte kullanılabilir(Anlamsal olarak birbirlerine yakın olduklarından değil, anlamsal ilişkileri olduğu için)
# ..

# Örn: Ben eve gidiyorum
# Ben işe gidiyorum
# Ben arabaya gidiyorum
# Model eve-işe-arabaya tokenlerinin anlamsal ilişkilerini yüksek buluyor.Ikame kullanım bunlar yani birbirlerinin yerine kullanılabilen

# Tokenler arasında anlamsal bir ilişki olup olmadığı şu şekilde hesaplanır

# class chat soru: Hocam bu değerler corpustaki bütün tokenlerin birbiri ile cosinus similarity değerleri değil mi?
# Johnson: Hayır hocam. Bu(alttaki tablodaki) değerlerin cosinus similarty ile aldığımız değerler ile alakası yok
# .. cosinüs similartiy uzaydaki vektörler arasındaki açıya bakarak bir sonuç çıkartacak(Alta bkz)

#!!! Örneğin Man-Woman arasındaki cosinus similarty hesabı; alttaki man ve woman 300 boyutlu bir feature representation
# .. üzerinden değerlerini almış olalım. man (1,300) lük bir vektör, woman aynı şekilde (1,300) lük vekttör
# .. bunların uzaydaki oluşturduğu vektörlerin açılarına bakarak cosinüs similarity hesaplanıyor
        Man    Woman
Gender  -1   -    1
Royal   0.01 -  0.02
Age     0.03 -  0.02
Food    0.09 -  0.01
...
...

"""
# Modelimiz feature representation ları arka planda belirliyor

# Elde ettiğimiz vektörleri uzayda görselleştirdimiz zaman
# cat ve kitten ın genelde aynı tokenlerle kullanıldığını bilmiş. Yani bunlar anlamsal olarak birbirine daha yakın
# cat ve dog un feature representation ı hayvandır, evcil olması, canlı olması olabilir. Bunlar arasında da bir anlamsal ilişki kurabilir(uzayda yakın olurlar)
# cat ile house arasında mesafe daha yüksek cat ve dog a göre. Yani cat ve dog arasında daha yüksek bir anlamsal ilişki kurmuş
# House a atadığı yüksek anlamda cat ve dog un evde yaşamasından dolayı bunlarla bir anlamsal ilişki kurmuş(düşük düzeyde)
# Man in woman ile olan anlamsal ilişkisi, King ile queen arasında olan anlamsal ilişkiye göre daha yakın
# .. yani man ile woman arasındaki anlamsal ilişkisi daha yüksek.
# .. Ancak dikkat edersek man king e daha yakın(Çünkü 2 side erkek) ve woman queen e daha yakın(Çünkü 2 side kadın)
# Buradaki anlamsal ilişkiler ile ilgili bilmemiz gereken;
# 1.Anlamsal ilişkiler DERKEN sözlükdeki anlamsal ilişkileri DEĞİL, birbirleriyle kullanım durumlarına göre olan durumu kastediyoruz
# 2.Anlamsal ilişkiler birbirlerine yakın olan tokenler ile daha yüksek olur
# NOT: BERT modelleri hariç diğer modellerde birbirlerine yakın olan tokenlerin anlamsal ilişkileri daha yakın

"""
Mesela bütün feature'lar 100 boyutlu olsun.  Man ve Woman 100 boyutlu vektörleri şu vektörlerle temsil edilsin.
.. (PCA'de birçok feature'ı .. temsil eden bir bileşen gibi)
.. Alttaki 2 vektör arasındaki açı 10 derece olsun

  Man     Woman
   |     /
   |    /
   |(10)/
   |  /
   | /
   |/

Bu açının cosinüs ü alınır. Cos10 olsun
Benzerlik belirlenirken arka planda cosinus similarity işlemi yapılır. aradaki açı ne kadar dar
.. ise anlamsal olarak birbirlerine o kadar yakınlar demektir. Aradaki açı 0 ise bu iki kelime
.. AYNI kelime demektir
"""
# Anlamsal ilişkili olan kelimelerin vektörel uzaklıklarının yanında öklid uzaklıkları da yakındır

# Cosine Similarity
# İki veya daha fazla vektör arasındaki benzerliği ölçer

"""
   y      A      B
   |     |     /
   |    |     /
   |   |(10)/               --> Cos(10) = 0.9848 --> %98 similar
   |  |   /
   | |  /
   ||/____________ x

Benzerliği sözlükteki anlama göre değil, birlikte kullanılma sıklığına göre veya aynı eylemi yapma
.. durumlarına göre kurar
"""

# Word Embedding Algorithms
# 1. Embedding Layer: Tokenlerimizi word embedding lere çevirmek için kullandığımız bir layer. DL derslerinde kullanacağız. Dl de konuşacağız
# 2. Word2 Vec
# 3. Global Vectors(Glove)
# 4. Embedding from Language Models(ELMO)
# 5. Bidirectional Encoder Representations from Transformers(BERT)

# Bir çok wordembedding var aslında. Ama biz Word2Vec ve Glove üzerinden bir şeyler yapacağız
# Elmo ve BERT de word embeddingleri belirlerken, word2vec ve glove a göre daha başarılı bir şekilde belirliyor
# Word2Vec içinde 2 farklı algoritma var

# Word2 Vec
# CBOW(Continuıus bag of words) vs skip-gram
# Word2Vec altındaki 2 algoritmanın çalışma mantığı ne
# Grammer yapısındaki kalıpları öğrenmeye çalışır. Örneğin Gmail de, whats app bir kelimeden sonra yazılan kelimeyi önerirken bu yöntem kullanılır
# Kırmızı spor araba trafik kazasına karıştı

# Skip-gram
# Örnek cümle: Kırmızı spor araba trafik kazasına karıştı
# modele input olarak orta kelimeyi verdiğimde sağındaki solundaki kelimeleri öğrenmeye çalışıyor(grammerde kullandığım kalıpları öğreniyor)
# windows_size=2 : Sağımda ve solumdaki kaç tokeni dikkate alarak anlamsal ilişki kurayım
# .. Orta kelime olarak "kırmızı"(w(t)) için w(t-2):boş, w(t-1):boş , w(t+1): spor, w(t+2) : araba olacak
# .. Orta kelime olarak "spor" için w(t-2):boş, w(t-1):kırmızı , w(t+1): araba, w(t+2) : trafik olur/olabilir...şeklinde anlamsal ilişkiler kurar
# Model bunları öğreniyor. # Class chat soru: token patternlerini öğreniyor diyebilir miyiz? # Johnson H: Evet
# Daha büyük pattern leri öğrenmesini istiyorsanız modeli windows_size arttırılır(Genelde 5 veya 10 verilir)
# Daha küçük datalarda tercih edilir

# CBOW
# Örnek cümle: Kırmızı spor araba trafik kazasına karıştı
# windows_size=2 için;
# modele input olarak kenar kelimeleri verip orta kelimeyi öğrenmeye çalışıyor(grammerde kullandığım kalıpları öğreniyor)
# w(t-2):Kırmızı, w(t-1):Spor ,w(t+1):trafik,w(t+2):Kaza olursa, orta kelime "araba"(w(t)) olur/olabilir...şeklinde anlamsal ilişkiler kurar
# Daha büyük datalarda tercih edilir

# NOT: windows_size ı arttırsanız bile yine de birbirine yakın olan kelimelere daha çok ağırlık verecektir model
# Class chat: CBOW: Bir bağlam verildiğinde ona en uygun sözcüğü bulma. Skip-gram: Bir kelime verildiğinde bağlamını bulma
# Johnson H: Aslında aynı şeyleri yapıyorlar

# Word2Vec vs Glove
# windows_size=2
# the nın quick ve the nın brown arasındaki ilişkileri kurup ve kalıpları öğreniyor(training de)
# .. anlamsal ilişkileri tokenlerin birbirleriyle sıklıkla kulanımlarımlarına göre kuruyordu
# .. örneğin quick ile the sıklıkla birlikte kullanılmıyorsa anlamsal ilişkileri düşük olacak
# .. örneğin quick ile brown sıklıkla birlikte kullanılıyorsa anlamsal ilişkileri yüksek olacak
# .. Burada tokenler birbirleriyle her karşılaştığında word embeddinglerini güncelliyor. O yüzden daha yavaş glove a göre
# Glove(2014-stanford üni) :Glove aslında Word2Vec alyapısını kullanıyor. Biraz daha farklı
# .. Word2Vec(google tarafından piyasaya sürüldü) de the ve quick 1 defa kullanıldı diyor ve  feature representation ı
# .. güncelledi sonra başka bir yerde görünce tekrar
# .. feature representation ını güncelliyor ve bunu 1000 defa görürse 1000 defa güncelliyor
# .. glove, the ve quick datada kaç defa kullanılmış not alıyor, sonra diğer iki kelimeye bakıyor onların sayısını
# .. tutuyor(bu istatistiği tutuyor) sonra güncellemeyi tek seferde yapıyor. Bu yüzden glove biraz daha kullanışlıdır
# NOT: word2vec leri modellere verirken yine glove ve word2vec de küçük harflere dönüştürmemiz gerekiyor

# ELMO ve BERT
# Elmo ve Bert in tercih edilme sebebi: # Glove ve word2vec sesteş ama anlamları farklı olan tokenleri tespit etmekte iyi değil. 
# .. Yani tek taraflı çalışıyorlar
# ilk cümlede Soldan sağa doğru model gider ve elmayı sevdiğimi görür sonra bir de sağdan sola gelerek derki
# .. bu marketten aldığımız elma
# Alttaki cümledede soldan sağa gider ve sağdan sola gider bu yediğimiz elmamı der sonra bakar bu o değil der
# .. ve bu 1. cümledeki the apple ile 2. the apple a farklı davranır. Glove ve word2vec bunlara aynı token muameleyi yapar

# %% NLP-4
##### Word2Vec
# WordEmbedding ten daha advance bir yöntemimiz yok(Sayısala dönüştürme metodları arasında)
# WordEmbbedding grammer yapısındaki patternleri öğreniyordu. Hangi tokenden sonra hangi token gelir
# .. bunları öğrenerek bizim konuştuğumuz textten anlamlar çıkarabiliyordu
# pip install gensim  # Anaconda kullanılmıyorsa bu indirilmeli

import warnings
warnings.filterwarnings("ignore")

from gensim.models import Word2Vec
# Zipfile daki dosyayı okutma yöntemi
text = zipfile.ZipFile('newspaper.zip')  # 'newspaper.zip': zip dosyamızın ismi
text = text.read('newspaper.txt') # 'newspaper.txt': zip dosyasının içindeki dosya
text = text.decode('utf-8') # Türkçe karakterleri okuyabilmesi için 'utf-8' kullanıyoruz. Çok fazla türkçe karakter olduğu için bunu yazmalıyız

# Zipten çıkartılıp dosya okutulmak istenirse;
# text = open('newspaper.txt', 'r', encoding='utf8')
# text = text.read()
# text[:1000]

text[:1000]
# Word2vec arkada bir RNN yani bir DL modeli kullandığı için cleaning yapılmış data vermeliyiz
# !!!DL modellerinde tokenization ve lower kesin olmalıydı, eğer isterseniz noktalama işaretlerinide bırakabilirsiniz DL de. Model bunları 
# .. anlamlandırabiliyor diğer adımlar gerekmiyordu cleaningte
# !!!ML de tüm adımlar yapılmalı(stopwords ve lemmatization dahil olmalı)

# Gazetelerden alinmis 400.000 cümleden olusan, cleaning islemi tamamlanmis bir datasetimiz var.
# Sadece tokenlerine ayıracağız bunu

# Word2vec de;
# 1.Datayı 2 boyutlu istiyor.
# 2.Her bir cümlem tokenlerine ayrılmış şekilde ve liste içerisinde olmalı

# NOT: Bazen encoding='utf8' bazen encoding='utf-8' şeklinde çalışıyor

print(text[:1000])  # new line lar( \n ) satırbaşları oldu

# Cümleleri birbirinden ayıracağım.(Bunlar new line lardan(\n) itibaren ayır)
# "\n" pattern ini gördüğün yerde gruplara böl/split et
list_sent = text.split('\n') # Bu pattern e göre text imi ayıracak
list_sent[:10]  # Liste içerisinde ilk 10 cümleyi görüyoruz
# Her bir cümlem string formatına dönüştü

list_sent[0].split() # İlk cümleyi seç ve boşluklardan itibaren ayır liste içinde topla # split in default u boşlu
# Her bir cümleyi tokenlerine ayrılmış şeklinde liste formatına dönüştürmüş olduk
# Bunu tüm cümlelere yapalım altta

corpus = []
for sent in list_sent:
    corpus.append(sent.split())

print(corpus[:10])  # Corpus un ilk 10 cümle.
# Her bir cümlem word tokenlerine ayrılmış liste formatına döndü
# Dikkat edersek 2 adet köşeli parantez var(Yani 2 boyutlu oldu)
# 2 şart da sağlanmış oldu. Modelimizi word2vec algoritmasına verip anlamsal ilişkiler kurmasını sağlayabiliriz

model = Word2Vec(corpus, vector_size=100, window=5, min_count=5, sg=1)
# corpus: 2 boyuta dönüştürdüğümüz corpus 
# vector_size=100 : (Not "vector_size" bazı sürümlerde "vector" şeklinde): word embbeding im kaç boyutlu olacak
# .. Best practice 50-100-300 rakamları ancak istediğiniz başka rakamlarda yazabilirsiniz
# window=5 : Tokenimizi sağındaki ve solundaki kaç tokeni dikkate alarak anlamsal ilişkiler kursun
# min_count=5 : Datamda 5 veya daha az geçen tokenleri dahil etme eğitime. Bunlardan bir anlamsal ilişkiler kuramaz(Çok bir şey katmaz eğitime)
# sg (Skip-gram) --> sg=0 olursa CBOW, sg=1 olursa Skip-gram demek. (Default=0)
# .. sg=1 : Skip-gram : orta kelimeyi ver yandakiler(5 tane)  ne olabilire bakacağız
# Bunu neden tercih ettik? Data küçükse skip-gram daha iyi sonuçlar veriyordu. O yüzden skip-gram kullandık burada
# NOT: Ortalama 5-10 dakika sürebiliyor bu kodun çalışması

model.wv['ankara']  # wv: word vector
# 100 boyutlu wordembedding i döndürdü
# Bu rakamların her biri bir feature ı temsil ediyor
# Burada farklı çıktılar görebilirsiniz

# Bakalım tokenler anlamsal ilişkileri iyi kurabilmiş mi?
model.wv.most_similar('öğretmen')
# most_similar: İçerisine verdiğimiz en tokene en fazla benzerlik gösteren tokenleri gösterir
# öğretmenin, öğretmeni: Burada ne benzer? Birbirlerine çok fazla anlamsal ilişki yüklemesinin sebebi benzer cümlelerde
# .. birbirlerinin yerine(ikame) kullanımlarının olması (Örn: öğretmen ödev verdi, öğretmeni ödev verdi, öğretmenin ödev verdi)
# 'okuldaki' : öğretmenden Önceki ve sonraki gelen tokenlerle kurduğu ilişki hasebiyle gelmiş(Yani DIKKAT:
# .. bunun benzerlik sebebi ile "öğretmenin" benzerlik gösterme sebebi farklı)
# 2 farklı şekil ile anlamsal ilişkileri yakalıyor 1.Birbirlerinin yerine(ikame)(Bu yüzden DL de lemmatization yapmıyoruz. Bunları yakalaması için)
# .. 2.Önceki ya da sonraki kelimelere göre

model.wv.most_similar('kırmızı')
# çizgileri: Gazete haberi olduğu için siyasi bir ifade ile anlamsal ilişki kurmuş(Yunanistanla kırmızı çizgilerimiz var)
# ışıkta: kırmızı ışık
# Sarı, turuncu : ikame kullanım(Yukardaki öğretmeni, öğretmenin açıklamasındaki)
# Corpus um kaliteli olsaydı "çizgileri" yerine "renk" gelmesini beklerdik ilk sırada

model.wv.most_similar('eve')
# evine : ikame kullanım

model.wv.most_similar('mavi')
# mavi marmara baskınıyla : Yani mavi-baskınıyla güçlü bir ilişki kurmuş(2 sonraki tokenle güçlü bir ilişki kurmuş)
# windows_size içindekilerle kurduğu anlamsal ilişkiler daha fazla diğerleriyle daha az yani window_Size
# .. dışında olanlarla da çok az da olsa anlamsal ilişkiler kuruyor

model.wv.most_similar(
    positive=['öğrenme', 'doktor'], negative=['tedavi'], topn=5)
# Hatırlatma : Wordembedding değerleri -1,+1 arasındaydı
# 2 tokeni toplayacağız burada.['öğrenme', 'doktor'] Yani;
# Örneğin : merhaba(a) bugün(b) nasılsın(c) --> Bunların hepsinin word embedding değerleri var(3 ünün de)
# .. bunları birleştirip bir sentence embedding elde ediyoruz.
# .. Bu cümleye asıl anlam katan "nasılsın". Çünkü nasıl olduğunu öğrenmeye çalışıyoruz.
# .. Bunları toplayıp sentence embedding yapacağız. Ancak wordembedding değerlerimiz -1, +1 arasında kalsın istiyoruz topladıktan sonra da.
# .. Bu yüzden bu kelimelerin önemine göre bir ağırlıkları olacak örneğin "nasılsın" a 0.8 ağırlık veriyor. Diğerlerine(merhaba, bugün)
# .. 0.1 ağırlık veriyor ve bu ağırlıklar ile word embeddingler çarpılarak yeni sentence embeddingler elde ediliyor
# .. Örneğin bunların wordembedding değerleri -1, -1, -1 şeklinde ise
# .. Sonuç : 0.8 * (-1) + 0.1 * (-1) + 0.1 * (-1) = -1 (Değerimiz -1 ile 1 arasında kaldı)

# Doktordan tedaviyi çıkar --> meslek oluşur --> öğrenmeyi ilave edersek alttaki 5 kelimeyi(topn=5) vermiş
# .. doktoru doktor yapan en iyi representation ı yani "tedavi" yi elinden aldım ve "meslek" tokeni kaldı
# .. "öğrenme" yi ilave ettim ve saçmasapan sonuçlar geldi. Demekki eğitim çok başarılı değil
# NOT: Çıktılar değişebilir

model.wv.most_similar(
    positive=['ankara', 'belçika'], negative=['brüksel'], topn=1)
# Belçikadan brükseli çıkar elimde "ülke" kalır ---> ankarayı eklersek
# .. bundan bir ülke(Danimarka) sonucu çıkarmış.
# 400.000 tane cümle ile bile yeterli öğrenememiş. Yani verimiz az aslında burada

# class chat soru: Hocam ankara ve belçikaya karşılık gelen vektörleri toplayıp brüksele karşılık gelen vektörü çıkardı değil mi?
# Johnson H: Aynen hocam

model.save("word2vec.model")             # Modeli kaydetme
model = Word2Vec.load("word2vec.model")  # Save edilen modeli çağırma

##### GLOVE
from gensim.models import KeyedVectors
# Hazır 6 milyar tokenle eğitilmiş(2014 teki wikipedia). Eğitim sonucunda elde edilmiş her biri 100 boyutlu embedding değerlerinden oluşuyor
glove_model = 'glove.6B.100d.txt'
model2 = KeyedVectors.load_word2vec_format(glove_model, no_header=True)  # glove_model yerine 'glove.6B.100d.txt' de yazılabilir.(Değişkene atamamız gerekmiyor illa)
# glove_model in farklı varyasyonları var. Internette bulabilirsiniz bu text dosyalarına
# glove modelden elde edilmiş Word embeddingleri "word2vec" formatına çevirmemiz gerekiyor. Bunu yapmamız lazım ki "wv" ile çıktılara bakabilelim
# .. yani tokenlerin diğer tokenlerle olan benzerliklerine bakabilelim
# KeyedVectors.load: Localimden yükle
# .. word2vec_format : word2vec formatında (yükle)
# no_header=True : Eski sürümlerde bunu yapmazsak hata alırız

model2['teacher']
# Glove modelin önceden eğitip tespit ettiği wordembedding değerleri olduğu için çıktılar herkeste aynı olacak

model2.most_similar('ankara')
# Burada kurulan anlamsal ilişkiler daha mantıklı(Word2Vec e göre). Ancak bunlar başarılı mı değil mi ? Bizim belirttiğimiz
# .. 2 tane tokenden bir token çıkarıp o anlamsal ilişkileri yakalayabilecek mi bakalım. Altta toplama çıkarma yaparak bakacağız
# istanbul: ikame kullanım
# moscow: komşu ülke olduğu için

model2.most_similar('teacher')

model2.most_similar('doctor')
# Buradaki sonuçlarda görüyoruz ki daha mantıklı ilişkiler yakalanmış bu veride
# Peki bu ilişkiler başarılı mı değil mi bakalım?

model2.most_similar(positive=['woman', 'brother'], negative=['man'], topn=1)
# Woman ile brother ı ağırlıklandırıp topluyoruz
# Brother dan man i çıkarırsam elimde sibling(kardeş) kalır buna "woman" ilave edersek "daughter" geldi

model2.most_similar(positive=['woman', 'father'], negative=['man'], topn=1)
# Father dan man i çıkar elimizde parents(ebeveyn) geldi buna woman eklersek mother gelmiş

model2.most_similar(positive=['woman', 'uncle'], negative=['man'], topn=1)

model2.most_similar(positive=['ankara', 'germany'], negative=['berlin'], topn=1)
# germany den berlin i çıkar --> country(ülke) kaldı buna ankara ekle --> ülke(turkey) gelmiş

model2.most_similar(positive=['teach', 'doctor'], negative=['treat'], topn=1)

model2.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
# king - man = royalty + woman = queen

model2.most_similar(positive=['love', 'jealous'], negative=['hate'], topn=1)

# Sonuç olarak 400.000 tane cümle az değil ama onda aldığımız ilişkilere bakın
# Birde bu veriye bakın. Sonuçlar bu veride çok daha başarılı

##### RNN MODELLER
# RNN: Tekrarlayan sinir ağları
# !!! Tokenler arasındaki anlamsal ilişkilerin kurulması. kalıpların tespit edilmesi vs DL modelleri ile yapılıyor
# Bir datanın içeriğinden ziyade o datanın sıralaması bizim için daha önemli ise bu tür datalarda kullanacağımız RNN modelleri kullanıyoruz
# Datanın sıralaması? Mesela ben türkçe dilindeki tokenleri sıralı şekilde kullanarak konuşuyorum. Eğer farklı kalıplar kullanarak konuşursam 
# .. anlaşılmaz. Modelin de bu grammer yapılarını anlamlandırabilmesi için sırası önemli

# ANN ve CNN modellerinde hafıza denen bir yapı yok yani kendini tekrarlayan bir yapı yok
# RNN de Her bir layer bir önceki layer ın çıktısına muhtaç. Peki burada neyi kastediyoruz
# Örnek: What time is it?
# Alttaki mavi RNN modelim ve bu tek bir layer
# !!! .. Hidden state modelin hafızasıdır.
# .. Ben what tokenini veriyorum t0 anında
# .. Sonraki aşamada yani modele t1 anında "time" tokenini verdiğimde what o bana geri gönderiyor(reccurent işlemi yapıyor) ve time ile what 
# .. birleşiyor. Eğer what ı hafızasında tutmasaydı "time" tek başına bir anlam ifade etmeyecekti. What ile time birlikte kullanıldıklarında
# .. bir anlam ifade ediyor o yüzden bizim hafızası olan bir modele ihtiyacım var. Devam edelim  t2 anında "is" i veriyorum ve "what time" ı 
# .. hafızada tutup ile "is" ile birleştirdi. t3 anından "it" i verdiğim zaman t4 anında modelim what time is it i birlikte gördüğü için bundan 
# .. bir anlam çıkarabilir
# What ile time geldi sentence embedding şekinde birleşiyor ve bir aktivasyon fonksiyonuna(tanh) giriyor
# .. Bu what time olarak çıkıyor sonra "is" gelecek. what time ile birleşecek
# !!!Ilk hidden state imiz 0 lardan oluşan bir word embeddingtir yani cümleye "what" ı verdiğimde "what" hala "what" olmaya devam edecek
# Bunları hafızasında tutmasaydı tokenleri tek tek değerlendirip bir anlam çıkaramayacaktı
# Yani her bir layer bir önceki layerdan gelen bilgiye mahkum derken bunu kastediyoruz
# Başka bir örnek: 10 tokenli bir cümlemizin olduğunu varsayıyoruz. Bunların herbiri 3 boyutlu embeddingler olsun t0,t1, ... t9 anında
# .. tokenler sırayla RNN modellerine veriliyor. Her bir model bir sonraki layer ı beslemeye devam ediyor

##### RNN i nerede kullanıyoruz
    # one to one: Çok kullanılmaz çünkü RNN model text "tek" iken oluşmaz genelde çoğunlukla birden fazla tokenden oluşur.
        # .. Bir kelime veririm "gülmek" .. Bana gülen bir resim verir
    # one to many: Bir resim veriyoruz modele, resimde ne görüyorsa onu vereceğimiz token sayısıyla anlatıyor(max 10 tokenle anlat vs)
    # many to one : Birden fazla tokenden oluşan text i veriyoruz, olumlu ve ya olumsuz diye döndürür
    # many to many : Translation olarak düşünebiliriz. Birden fazla tokenden oluşan text i veriyoruz Birden fazla tokenden oluşan text geliyor
# NOT: Bunların hepsini hazır modeller üzerinden göstereceğiz

##### !!!Ne için Tanh activasyonunu kullanıyoruz?
    # Bir nevi scale diyebiliriz
    # Modelimiz RNN modelleri içerisinde sürekli işleme tabi tutulurken katsayılarla(1 den büyük değerlerle) çarpılıyor çokça defa.(Girerken, 
    # .. başka bir tokenle birleşirken vs vs) değerlerimiz çok büyük değerler alıyor(-1, +1 arasında olmuyor. Bunlar -1, +1 arasında kalmalı ki 
    # .. anlamsal ilişkileri kurabilsin model)
    # Örneğin Man tokeninin word embeddingleri beklediğimiz tarzda oluyor diyelim tanh kullanmadığım halde istediğim değer aralığında olsun(-1,1) 
    # .. man = [1, 0.2, 0.3, 0.8, ...]  , ve woman = [100, 10, 0.001, 0.003 ...] olsun
    # man ile woman arasındaki benzerliği biz gender arasından buluyorduk ama böyle olursa değerler birbirine yakın olmadığı 
    # .. için(1-100 , 0.2-10, ...) anlamsal ilişki kuramaz model. Yani bu değerleri -1, +1 arasına sıkıştırmalıyız. Bu yüzden tanh kullandık

# class chat soru: 100 boyutlu derken 100 token feature olarak kullanılıyor mu demek?
# Johnson H: Hayır hocam. Bu değerler bir feature representation, Gender, Araç, Royalty vs vs
# .. Bu feature representation ların ne olduğunu hiç bir zaman bilmiyoruz. Model bu anlamsal ilişkileri yakalıyor arka planda ancak bana bir isim 
# .. vermiyor. Buradan her token için benzerliği 1-2 tanesi üzerinden kuruyor(hepsini kullanmıyor)
# class chat soru: 100 feature tüm token ler için aynı mı
# Johnson H : Evet aynı, ama hepsinin aldığı sayısal değer farklı

##### RNN - Sequential Memory: Sıralı hafıza
# RNN modelleri sürekli kullandığımız patternlere/grammer yapıların aşina oluyor ve sıralı hafızaya sahip oluyorlar
# .. Size desem ki alphabeyi sayın desem sıradan a dan z ye hemen sayarsınız. Z den a ya sayın desem zorlanırsınız
# .. Çünkü o yapıya aşina değilsiniz.
# Bizim modelimizde ne kadar bu kalıplara maruz kalırsa bu yapıları Sequential Memory sayesinde tespit edip anlamlandırabiliyor
# "I met my best friend yesterday" yerine "Best yesterday I friend my met" desek hiç kimse bir şey anlayamaz

# Modelim en sonunda text in bütününü anlamlandırıp, anlam çıkarıp sonuç çıkarıyor ancak her şey güllük gülistanlık değil
# Burada 2 problem var
# 1.Sönümlenen gradient(Vanishing)
# 2.Patlayan gradient(Exploding)
# Bunlar unutma problemidir(İkiside) Not: Exploding gradient ile çok nadir karşılaşırsınız
# Siyah yeri RNN modelin hafızası olarak düşünelim. what %100 yer alıyor(%100 siyah)
# .. Siyah yeşil yer: What da %50 lik unutma oluyor, what ve time ile hafıza paylaşıyor.t2 anında(O3) : "is" in hafızası daha fazla diğerleri az
# .. en sona en önemli olan "what" ve "time" tokenleri neredeyse hafızada çok az kalmış. RNN modellerinin böyle bir sorunu var
# RNN kısa cümlelerde çok iyi sonuçlar verirken, uzun cümlelerde çuvallıyorlar.
# Bunları çözmek için yeni bir model geliştiriliyor ve unutma problemi gideriliyor(LSTM ve GRU ile)
# class chat: unutmanın başta olması bizimki gibi fiili sonda olan diller için bir avantaj olabilir mi hocam?
# Johnson H: Yorumu en başta yaparsanız sonlarda asıl konuyu unutmaya meyilli olacak
# .. Eğer yorumu sonda yaparsanız bir avantaj olabilir ama çok uzun cümlelerde yine çuvallayacaktır
# RNN modellerini hiç kullanmayacağız zaten piyasada da kullanımları yok LSTM ve GRU kullanacağız

# Kısa süreli hafıza problemi vanishing gradient probleminden kaynaklanıyor(kaynaklarda böyle geçer) ancak unutma exploding gradientden de kaynaklanır.
# !!! NOT: Gradient: Katsayılarda ne kadar güncelleme yapacağımıza karar verdiğimiz bir argüman
    # Gradient değerimiz ne kadar büyükse, güncellemeyeler de büyük olacak yani eğim o kadar dik olacak ve adımları büyük büyük atacak
    # Gradient değerimiz çok küçük olursa, eğitim neredeyse biter eğitim devam etme bile bir şey değişmeyecek
    # Gradient değerim ne kadar büyük olursa backpropagation da adımlarım o kadar büyük oluyor(Learning rate le ayarlama yapabiliyorum ama
    # .. learning rate çok etki etmiyor). Sonuç olarak Katsayıların ne kadar güncellenecek bunu ayarlıyorve hangi feature ın katsayısı daha büyükse
    # .. o katsayı daha çok güncelleniyordu
# !!! Vanishing gradient;
    # Geriye doğru kısmı türevler alırken gradient değerlerimiz o kadar küçük ki;
    # Örneğin 10. epoch da katsayım 1.01 iken 100. epoch da katsayım 1.0098 ise. Yani katsayım çok değişmezse loss değerimde değişiklik olmaz. 
    # .. Demekki gradient değerim o kadar küçük bir değer ki eğitim devam etmesine rağmen model bir şey öğrenmiyor. Yani, vanishing gradient oluyor
# !!! Exploding Gradient;
    # Geriye doğru kısmı türevler alırken gradient değerlerimiz o kadar büyüyor ki;
    # Örneğin değerim ilkinde 1. epochda katsayım 100 iken, 100. epoch da 10 üzeri 100 oluyor
    # .. o kadar büyük değer geliyor ki katsayım loss değerleri "None" gelir. Bu durum exploding gradientti
    # Bu durum klasik RNN modelleri kullandığınızda olur biz bunları kullanmayacağız zaten.
##### Vanishing gradient ve exploding gradient neyden kaynaklanıyor. 
    # Modele token hepsinin çarpıldığı katsayı aynıdır. 
    # .. Loss hesapladık diyelim Biz geriye doğru kısmi türev alırken her layerdaki katsayılar birbirleriyle geriye doğru çarpılıyordu
    # Geriye doğru kısmi türev alırken her bir aşamadaki(layerdaki) bütün katsayılar birbirleriyle çarpılır 4 token için backpropagation aşamasında 
    # .. katsayı w üzeri 4 değeri alacak
    # Değerler 0-1 arasında bir değer olursa üzeri alındıkça 0 a yaklaşır katsayı     (Vanishing gradient)
    # Değerler 1 den büyük olursa Değer artı sonsuza gider(Giderek üstel olarak büyür) (Exploding gradient)
        # Kısa cümlelerde çok sorun olmaz ama uzun cümlelerde mesela 100 token olursa w üzeri 100 olacak ve türev alma 
        # .. aşamasında(güncelleme yaparken) sıkıntı yaşar model

##### Long short-term memory - LSTM
# RNN modellerinde unutma problemi olduğundan LSTM modeli bulunuyor(1995-1997 arası)
# Kısa süreli hafıza problemini gidermek için kurulan bir model
# 3 tane kapısı var : Forget gate, input gate, output gate
# Bu kapılar sigmoid fonk. üzerinden yapılıyor
# Buradaki hidden state, RNN deki hidden state ile aynıdır. Burada, kısa süreli hafıza olarak geçer
# !!! Cell state ve hidden state tamamen aynı mekanizmadır: uzun süreli hafıza
# !!! Hidden state, cümledeki veya textteki bütün tokenleri hafızada tutmaya çalışır
# !!! Cell state, modeldeki tahminleri yapmak için gerekli olan keyword leri öğrenip ayıklayıp tutuyor. Yani
# .. mekanizma aynı ama buna uzun süreli hafıza denmesinin tek sebebi sadece ilgili keyword leri tutmasıdır
# 100 tokenlik yorum için hidden da 100 token yer alırken, cell state de sadece keyworler yer alır.
# Yorumu tahmin etmemdeki en yüksek/en iyi token ne ise onu tutuyor cell state.

##### Keyword leri hidden state den alıp nasıl cell state e aktarıyoruz?
    # Yani cell state için keyword leri biz  Forget gate, input gate, output gate yardımı ile ayıklıyoruz
    # cell state e bir keyword değerinin geçmesini istiyorsam aktivasyon fonksiyonu(sigmoid) yardımıyla wordembedding i 1 ile çarptığımız zaman 
    # .. uzun süreli hafızaya(cell state e) geçiyor.word embedding i 0 ile çarptığımızda ihmal ediyor ve o cell state e geçemez
    # Yani, sigmoid function word embeddingleri 0 la veya 1 ile çarpılıyor

# Forget Gate: Cell state de bulunan mevcut bir bilgiyi unutturur. Bilgi yoksa bir işlem yapmaz
    # Örnek yorum: "Ilk aldığımda telefon müthişti ancak ilerleyen zamanlarda telefon çok kötü oldu"
    # Modelim eğitimi tamamlamış. Eğitim sonrasında hangi keyword lerin olumlu ve olumsuz yorumları tespitinde önemli olduğunu biliyor diyelim
    #..  "Ilk" kelimesini verelim.(Bu 0 lardan beslenecek(Yukarda bahsetmiştik)). Burada forget gate bir şey yapmayacak
    # .. çünkü daha cell state de hiç bir şey yok
# Input gate: Bir tokenin cell state e geçip geçmeyeceğine karar veriyor
    # Örneğin "ilk" kelimesi geldi 0 ile çarptı ve cell state e geçirtmedi diyelim
    # .. "aldığımda" kelimesini verelim ...... 0 la çarpıyor ..... (Forget gate hala bir şey yapmıyor)
    # .. "telefon" kelimesi geliyor. Input gate diyorki bu bir keyword ve bunu sigmoid vasıtasıyla 1 ile çarpıyor sonra
    # .. bir de birçok katsayı ile çarpma vs olduğu için tanh den geçiriyoruz
    # .. ve değerlerimizi -1 ile +1 arasında sıkıştırıyoruz. cell state e aktarıyoruz
    # "müthişti" ...... "telefon" ile aynı işlem yapılıyor ve cell state e aktarıyoruz. cell state de şu an "telefon" ve "müthişti" var
    # "ancak" .... 0 ile çarpıyor
    # "ilerleyen" .... 0 ile çarpıyor
    # "zamanlarda"  ... 0 ile çarpıyor
    # "telefon" . bir keyword geliyor forget gate bir şey yapmıyor ancak input gate diyorki ben de "telefon" mevcut zaten
    # .. bunu 0 ile çarpıp ignore ediyor.
    # "çok kötü" geldiğinde "müthiş" i silmesi gerekiyor forget gate ve "müthiş" in word embedding değerini sigmoid i 0 yapıp onunla
    # .. çarpıp çıkartıyor
    # modelim  en son "telefon çok kötü" üzerinden tahmin yapacak

# class chat soru: “telefon kesinlikle bu parayı hak etmiyor. yeni hiç bir özelliği yok. ancak küçük ekran sevenler için iyi bir telefon”
# .. şeklindeki bi yorumu yanlış anlamaz mı hocam bu mantıkla?
# johnson H: Yorumlarınızda böyle modelin kafasını karıştıran model varsa iyi örnekler üzerinden tahmin yapılmaya başlayacaktır ancak bu örnekler 
# .. daha çok verilirse model daha iyi öğrenir. Bunu modele öğretmeniz lazım örneğin; Open AI firmasının geliştirdiği "gpt 3" modeli gibi çok 
# .. detaylı bilgileri olan bir corpusunuz olursa sonuçlarınız o kadar iyi olur

# %% NLP-5
# ÖZET:
# Input gate : Cell state e bilgileri ilave etmek için kullanılır. Bunu sigmoid ile yapıyor. 1 ya da 0 la çarpıyorduk
# .. 1 le çarparsak bilgi kalıyordu. 0 ise bilgi cell state e geçmiyordu
# Output gate: Bir sonraki hidden state imizi belirliyor.  Output gate her "t" anında çalışır. Her t anından hidden state ve cell state 
# .. güncellenmeye devam eder. Biz tahminlerimizi cell state üzerinden değil hidden state üzerinden yapıyoruz. Ancak modelim tahminleri yaparken 
# .. cell state deki keyword lere göre yapıyor
# .. O zaman biz cell state i hidden state e eşitlemem lazım ki cell state deki ilgili keyword ler
# .. üzerinden hidden state tahminlerini yapsın
# Forget gate in bir işlem yapabilmesi için cell state de bir bilgi olsun ki onu unutturmak için kullanılsın
# class chat: output gate bir kez mi çalışıyor
# Johnson H: Hayır. Output gate her "t" anında çalışır. Her t anından hidden state ve cell state güncellenmeye devam eder
# .. Cell state her iterasyonda başa döner(recurrent)(mükemmel eğer key word olduysa bunu başa atar yine mükemmel ile
# .. "tv" tokenleri birleşecek vs) ve her iterasyonda bilgi hidden state e aktarılır.
# !!! NOT: layerların birbirini besleme olayı yani recurrent olayı LSTM ve GRU da da devam ediyor

##### Gated Recurrent Unit(GRU)-2014
# LSTM ile çalışma farklılıkları var. GRU seri/hızlı çalışır. Bu daha çok tercih edilir ama ikisini de deneyin hangisi daha iyi sonuç verirse 
# .. onu kullanın. Tahminleme arasında fark yok

# LSTM de hidden state ve cell state in mekanizmaları ayrı iken. GRU da tek bir mekanizma üzerinde birleşir(bu yüzden daha seri çalışır)
# reset gate  : Hidden state(hidden state ve cell state tek mekanizma olduğu için) üzerindeki gereksiz bilgileri silme işlemini reset gate ile 
# .. yapıyoruz
# update gate : Bir bilginin güncellenmesi veya (en fazla)1 veya 2 tokenin unutturulmasını sağlar(forget gate gibi çalışır)
# .. bir tokeni kaldırıp yerine diğer tokeni getiriyor

# Working logic of RNN
    # Örnek: Yemeğin hangi dünya mutfağına ait olduğunu bilen bir modelimiz var diyelim
    # RNN lerin hafızasında 2 token tutabildiğini varsayalım(Normalde ne kadar token tuttukları bilgisi hiç bir dökümanda yazmaz)
    # Şimdi bunu RNN ile yapmaya çalışalım
    # Cümle: Dhaval eats samosa almost everyday, it shouldn't ve hard to guess that his favorite cuisine is Indian
    # Dhaval ı verdik hafızada tuttu, sonra eats i verdik "dhaval" ve "eats" i hafızada tuttu, 
    # .. sonra samosa gelince Dhaval gitti "eats ve samosa" yı aklında tuttu vs vs
    # . . bu şekilde devam ediyor en son da is geldiğinde elimizde "cuisine" ve "is" kalıyor. 
    # .. Ancak modelin tahmin yapabilmesi için hafızasında "samosa" tokeni olması lazım ki "india" şeklinde tahmin edebilsin
    # RNN modelleri "samosayı" unutmuş oldu ve tahmin yapamadı

# Working logic of LSTM and GRU
    # Cümle: Dhaval eats samosa almost everyday, it shouldn't ve hard to guess that his favorite cuisine is Indian. His brother
    # .. Bhavin however is a lover of pastas and cheese that means Bhavin's favorite cuisine is Italian
    # Burada da hafıza da 2 token tutulduğunu varsayalım
    # NOT: hidden state hafızasında 2 token tutabiliyorsa, cell state de 2 token tutar(Yani aynı sayıda)
    # .. Hidden state ve cell state in max ne kadar tutabildiğini bilemeyiz net olarak. Burada örnek olması için "2 token tutabiliyor olsun" denildi
    # Dhaval verildi. Bu bir keyword değil dedi ve aktarılmadı cell state e
    # Eats verildi  ....................... aktarılmadı
    # samosa                      .....     aktarıldı
    # hidden state tüm bilgileri tutarken cell state hafızasında sadece "samosa" bilgisini tutar ve tahmin olarak "indian" der model
    # Cümleye devam edersek , samosayı siler en son "pasta" ve "cheese" kalır ve "italian" ı tahmin eder
# Class chat soru : keywords leri nasıl belirliyor
# Johnson H : Keyword ler eğitim aşamasında belirleniyor

# Hocam ne zaman GPU veya TPU kullanmak gerekli?
# NOT: TPU , GPU nun daha hızlı halidir. Bert de 1 epoch 2 saat sürüyorken TPU 9 dk sürer. TPU sadece tensorflow kütüphanesinde çalışır

##### NLP with DL
# from google.colab import drive
# drive.mount('/content/drive')

# NOT colab da --> düzenle --> not defteri ayarları --> GPU seçin --> kaydedin.(Seçili olsa bile kaydede basın)

df = pd.read_csv('/content/drive/MyDrive/hepsiburada.zip')
"""
import zipfile

# Unzip the file
zip_ref = zipfile.ZipFile("/content/drive/MyDrive/ML_NLP/hepsiburada.zip", "r")
zip_ref.extractall()
zip_ref.close()
"""
df.head() # hepsi buradadan çekilmiş olumlu ve olumsuz yorumlarımız var

# NOT: NLP de missing values lar doldurulmaz. Drop edilir. Boş bir text görürsek mutlaka silinir.
# .. Bazen yorum yapmaz ama olumlu sütununa 1 verir. Onları olumlu tokenlerle doldururlar bazı notebooklarda
# .. ama bu gereksizdir zaten o tokenlerin olumlu olduğunu model biliyor
df.info()
# Olumlular 1, olumsuzlar 0

df.Rating = df.Rating.map({1: 0, 0: 1})
# hedefim benim negatifler olduğu için 1 leri 0 , 0 ları 1 yaparak mapledik
# !!! Çünkü mesela ANN de sonuçlar gelirken(örneğin: recall, loss) bu sonuç "1" label ının sonucudur. O yüzden değiştirdik
# .. multiclass larda ise "recall" yazarsanız modeliniz "micro recall" u yani ortalamaları takip eder. Aslında bir nevi accuracy
# .. micro recall = micro precision = micro accuracy --> Bunlar eşittir yani hepsi aynı skordur. O yüzden datanız multiclass ise
# .. oraya recall ya da accuracy vs yazmanız farketmez. Bu bilgiyi okuduğunuz makalelerde görebilirsiniz

df.Rating.value_counts() 
df.Rating.value_counts(normalize=True) # inbalanced %94,3  olumlu , %5,6 olumsuz yorumlar

# Tokenization
X = df['Review']
y = df['Rating']


num_words = 15000 # corpusta geçen en fazla kullanılan ilk 15.000 kelimeyi(tokeni) alacağız gerisini yok sayacağız.
tokenizer = Tokenizer(num_words=num_words)
# num_words=15000 : Tokenleştirme işlemi yaparken(Textlerin sayısal forma dönüştürürken) eğitimin en fazla kaç tokenle yapılacağını belirleme
# .. sadece datamda en sık kullanılan ilk 15000 tokeni dikkate al ve eğitimi yaparken bu 15000 tokeni dikkate al demek
# .. En uygunu bütün tokenlerin kullanılması ancak hesaplama maliyeti olacaktır. 
# .. Makinanız güçlü değilse 10000 veya 15000 deneyin(best practice(classification ve sentiment analiz için bu rakamların eğitim için yeterli olduğu görülmiş))
# .. Ancak datadaki tüm tokenleri kullanmak yerine bunu kullanmanın skorlarınızda az da olsa düşme oluşturacağını unutmayalım
# NOT: num_words= None --> None olursa bütün tokenleri kullan demek.
# Burada tokenlerine ayırırken küçük harfe dönüştürüyor, noktalama işaretlerini kaldırıyor
# filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'  : Default u bu. içindeki bu karakterleri temizliyor ama
# ..  Eğer sayıları da temizlemek istiyorsak "Tokenizer" içine filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n1234567890' yazabiliriz
# lower=True : hepsini küçük harflere dönüştürüyor # default
# split = ' ' : boşluklardan itibaren tokenlerine ayırıyor
# !!! Yani burada 1.noktalama işaretlerinden temizliyor 2. küçük harfe dönüştürüyor 3.tokenlerine ayırıyor(cleaning bitti)
# class chat soru: karşılaşmadığı kelimelere ne işlem uyguluyor?
# Johnson H : Onları ignore ediyor
# class chat soru: Burada oran şeklinde verebilir miyiz?
# Johnson H : num_words = the maximum number of words to keep, based on word frequency.
# .. dökümanında geçen açıklaması bir oran şeklinde belirtilmemiş. the max number of words ifadesi zaten integer sayıyı tarif ediyor hocam.

tokenizer.fit_on_texts(X)  # tokenizer işlemini X e(yorumlarıma) uygulla

tokenizer.word_index # Datamda en sık kullanılan kelimeleri kullanım sıklıklarına göre sıraladı
# Datamda en sık kullanılan kelime "çok" muş. # output:  'çok':1 , 'bir': 2, 've' : 3, 'ürün':4 .....
# Text i sayısal forma dönüştürürken bu rakamları kullanacağım. Örneğin "bir" kelimesi sayısal olarak "2" ye dönüşecek

len(tokenizer.word_index)  # Unique token sayısı # 217983 
# Şimdi eğitimin ilk 15000(num_words=15000) tokenle yapılacağını söyledik ama tokenizer ın uygulandığı token sayısı 217983 .. Bu ne anlama geliyor?
# Text imde geçenlerin hepsine token muamelesi yapar. Tokenlerine ayrıştırır ancak "tokenizer" değişkenimizin için hem eğitimi  hem dönüşümde
# .. ilk 15000 ini ile yapacağı bilgisi hafızasında

##### Converting tokens to numeric
X_num_tokens = tokenizer.texts_to_sequences(X) 
# texts_to_sequences : text i sayısal dizine dönüştür
# Bütün yorumlarım sayısal forma dönüştü
# Kontrol yapalım altta

X[105]  # Yorumumum bu. Bu yorumun sayısal hale dönmüş haline bakalım altta

print(X_num_tokens[105])
# Bakın 5. kelime "çok" du. Bu "1" e dönüşmüş. Çünkü üstte "çok" un karşısında 1 vardı
# !!! Ancak üstte(X[105]  kodunda) "çok" üstteki text te 6. sırada ama bu çıktıda "1" numarası 5. sırada. çünkü
# .. üstteki textte ilk 6 kelimeden(Logitech ürünlerinin geneli kalite olarak çok) 1 tanesi datamda en çok kullanılan 15000 kelimeden biri değilmiş
# .. dönüşüm sadece ilk 15000 tokene göre yapıldı

# class chat soru:  niye stopwords  yapmadık  hocam? ilk 15000 de bu kelimeler çokça geçiyor
# !!! Johnson H : SAdece ML de stopwords çıkarılıyor. DL de stopwords ler çıkarılmaz çünkü hangi token hangi
# .. tokenle anlamsal ilişkiler kuruyor buna bakacağız. Yoksa model öğrenemez

##### Maximum number of tokens for all documents
len(X_num_tokens[105])  # 105. yorum 22 tokenden oluşuyor
len(X_num_tokens[106])  # 106. yorum 18 tokenden oluşuyor
len(X_num_tokens[6150]) # 6150. yorum 88 tokenden oluşuyor
# DL modellerimiz, ML modellerimizde olduğu gibi datanın aynı boyutta olmasını ister. Bunların boyutlarını sabit aynı boyuta getirmeliyiz
# .. Çünkü bir yorum 22 tokenden oluşuyorsa(len(X_num_tokens[105])) diğeri 18 tokenden oluşuyorsa(len(X_num_tokens[106]))
# .. bu hata verir ancak bu boyut sayısını nasıl belirleyeceğiz ona bakalım

num_tokens = [len(tokens) for tokens in X_num_tokens] 
num_tokens = np.array(num_tokens) # nümeriğe dönüştürdüğüm yorumlar içerisindeki her bir yorum kaç tokenden oluşuyor bakıp array e dönüştüreceğiz
num_tokens # 1. yorumum 4 tokenden oluşuyor ... 3. yorumum 66 tokenden oluşuyor. Bunların hepsini eşit uzunlukta vermem lazım modele
num_tokens.mean()  # Ortalama 21 tokenden oluşuyor yorumlar
num_tokens.max()   # En uzun yorum 298 tokenden oluşuyormuş.
# .. Bütün yorumlarımı 298 e sabitlersem hiç bir bilgi kaybı olmaz. Eğer daha az sayı verirsek, mesela 100
# .. 298 tokenden oluşan bir yorumdan 198 tane token kırpılacak yani bilgi kaybı olur
# .. O yüzden en uzun olan token sayısına sabitlersem yorumlarımı bilgi kaybımız olmaz
#num_tokens.argmax()      # listedeki en uzun yorumun indexini argmax ile bulabiliyoruz.
#X[21941]                 # En uzun yorumum bu
#len(X_num_tokens[21941]) # 298

"""
# .. Peki sayıyı nasıl belirleyeceğiz eğer bazı yorumlarda bilgi kaybı olacaksa...
# Burada yapmamız gereken işlem ne altta örnekle anlatalım
# 6 yorumum var. 1. yorumum 5 tokenden oluşuyor .... 6. yorumum 15 tokenden oluşuyor olsun
list = [5, 10, 8, 9, 12, 15]
# Bütün yorumlarımı 11 tokene sabitlersem. 5. yorumumda 1 token, 6. yorumumda 4 yoken kırpılacak(O yüzden onlar "False" geldi çıktıda)
print(np.array(list) < 11)
# True ları topladı # Corpusunda 6 yorum var. Eğer bunları 11 tokene sabitlersen 4 tanesinde bilgi kaybı olmayacak anlamına geliyor
print(sum(np.array(list) < 11))
# Bulduğum 4 sayısını datamdaki tüm yorum sayısına oranlıyoruz 4/6. Yani corpusumun %67 sinde bilgi kaybı olmayacak
print(sum(np.array(list) < 11)/len(list))
# Biz bu oranın %95 olmasını istiyoruz best practice olarak. %5 lik kısmında bilgi kaybı olmasını göze alacağız(Çalışma maliyetinden kısmak için)
# Şimdi bu işlemin aynısını datamıza uygulayalım altta
"""
max_tokens = 61 # Bu sayıyı kendimiz belirliyoruz. Hoca önceden belirlemiş burayı
# sum(num_tokens < max_tokens)
# len(num_tokens)
sum(num_tokens < max_tokens) / len(num_tokens)
# Bütün yorumlarımı 61 tokene sabitlersem datamın %96 sında bilgi kaybı olmayacak. %4 hatam olabilir ama bu hatayı çalışma maliyetini azaltmak için göze alıyoruz
# num_tokens: Her bir yorumum kaç tokenden oluşuyor. Bunu gösteriyordu

##### Fixing token counts of all documents (pad_sequences)
# Şimdi yorumlarımızın hepsini aynı boyuta getirelim
X_pad = pad_sequences(X_num_tokens, maxlen=max_tokens)
# pad_sequences : doldurma işlemi
# Datamdaki textlerimizi tek bir forma sabitleme işlemi yapıyoruz
# maxlen = max_tokens = 61 : Hangi boyuta sabitlemek istiyorsak
# maxlen = None yazılırsa en uzun yoruma göre sabitler(bütün yorumları 298 e sabitler) 
# .. Ancak maxlen = 61 yazılırsa yani integer girdiğinizde hata alırsınız. 61 i bir değişkene eşitleyip öyle vermeliyiz maxlen e
X_pad.shape # 243497, 61

"""
# Padding mantığı
# Token sayısı 61 den az olanlar için padding mantığı
    np.array(X_num_tokens[800])  # Orjinal hali(61 den kısa bir yorum)
    len(np.array(X_num_tokens[800])) # 31
    X_pad[800]  # padding işlemi uygulanmış hali # Başına 30 tane 0 ilave etti başa(Doldurma işlemi yaptı)

# Token sayısı 61 den fazla olanlar için padding mantığı
    np.array(X_num_tokens[21941])  # Orjinal hali(61 den uzun bir yorum)
    len(np.array(X_num_tokens[21941])) # 298
    X_pad[21941]  
#!!! Cümlede ana fikir genelde cümle sonunda olduğu için sondaki 61 tanesini alıyor baştakileri ignore etti
"""

##### Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, stratify=y, random_state=0)
model = Sequential() # Ben bir DL modeli oluşturuyorum demek, Sıralı(sequential) şekilde layerlarımı oluşturabilirim demek(LSTM, GRU, dropoutlarını vs hepsini buraya ilave edebilirim)
embedding_size = 50

model.add(Embedding(input_dim=num_words, output_dim=embedding_size, input_length=max_tokens))    
# Embedding: Embedding layer diye yeni bir layer koyduk. Bu DL modellerinde kullandığımız kendine has bir metoddur
# .. Yani burada word2vec, glove vs gibi bir metod kullanmıyoruz
# .. Burada 3 tane parametrem var. Bu 3 bilgiyi kesinlikle istiyor
# input_dim=num_words    : Giriş boyutu=15000    : Eğitim için girecek token sayısı Eğitim de datamda en fazla geçen ilk 15000 tokeni kullan
# embedding_size         : Çıkış boyutu=50       : Dönüştüreceğim word_embeddingler kaç boyutlu olsun
# input_length=max_tokens: inputumun uzunluğu=61 : Dataya girecek olan her bir yorumun sabit uzunluğu ne? 
# NOT: num_words yerine integer bir sayı yazarsanız hata alırsınız. 15000 i mutlaka bir değişkene eşitlemeniz gerekiyor
"""
                       1    2     3 ....................... 61  -->  inputumun uzunluğu
                     (32) (2525) (14934)                        --> eğitilen tokenlerin numarası(Buradaki sayı max 15000 olabilir)
                                                                -- > soldaki(y ekseni) de 50 boyutlu word embeddingler
word embeddings(50) 1 0,31  0,43
                    2 -0,22
                    3 0,45
                    ...
                   ...
                   50
"""
# Word embeddingler ilk başta random atanacak. word embeddinglere dönüştükten sonra GRU modeline aktarılacak

model.add(Dropout(0.2)) # Dropout yerine batch_normalization da denenebilir. Hoca denemiş iyi sonuç gelmemiş
# 50 boyutlu word embbedingler GRU ya aktarılırken 0.2(maximum %20) si sönümlenecek dropout da
#!!! Not: Nöronlara uygulanan sönümlendirme maximum %20 dir. Her zaman %20 uygulamaz

model.add(GRU(units=48, return_sequences=True)) # Boyutu 50 ile geldi, nöron sayısını 48 e düşürdük. 
# units=48: 1.nöron sayısı ya da 2.bazı makalelerde: "yeni word embedding boyutu" olarak da geçer. 2 side doğrudur
# return_sequences=True: Bunu True yapmazsanız modelleriniz çalışmaz. Çünkü t0 anındaki hidden state im "what" t1 de "what time"
# .. t4 anında "what time is it ?" . Burada "is" ve "it" i  keyword olarak düşünelim(What ve time önceden keyword dü)
# .. Eğer cell state de çok fazla keyword birikirse cell state de de unutma durumu oluşabiliyor. Bir sonraki lstm layer ına
# .. cell state deki bilgileri aktarırken(Aslında hidden state ler aktarılıyor çünkü her aşamada hidden state ve cell state ler eşitleniyordu)
# .. t4 anındaki bilgileri gönderirken ilk olanlar(What ve time) unutulabilir. return_sequences, her t anındaki
# .. hidden state leri diğer layer a gönderiyor. Yani eğer olurda son t anındaki hidden state bilgilerini unutursa onlara önceki layer lara bakıp hatırlayabiliyor
# .. sonraki layer "dense layer" olana kadar return_sequences=True yapıyoruz. 
# return_sequences=False : Sadece son layerdaki(son t anındaki) bilginin gönderilmesi # Ancak zaten bu hata verir
model.add(Dropout(0.2))
model.add(GRU(units=24, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(units=12))
model.add(Dense(1, activation='sigmoid'))
# .. Tahminin yapılacağı Dense de son t anındaki hidden state aktarılıyor. Son hidden state imde çok fazla bilgi varsa tahminlerini yaparken
# .. ona göre yapacak yani unutmalar olabilir(Dense layer da geriye dönemiyor. Orada unutma mecburen olacak)
# .. Yani son t anındaki hidden state imiz unutmaya açık bir hidden state ise dense layer da bu unutma mecburen olacak
# .. çünkü tahminlemeyi dense layer da yaparız
# Eğer problem multi class olsaydı Dense(1, activation='sigmoid') için: "1" yerine target sayımız kaç ise onu yazacaktık activation = 'sotfmax' olacaktı

# class chat : input_length farklı gelirse ignore ediyor?
# Johnson H : Hata alırsınız(Mesela 61 yerine 62 dersek hata alırız)
# class chat soru : embedding size ı neye göre belirledik hocam
# Johnson H: 50, 100, 300 best practice dir. Özellikle NLP algoritmalarında model çok kolay overfit e gidebiliyor.
# .. Bunu engellemek için dropout oranlarıyla oynanabilir(0.5 e kadar(best practice) deneyebilirsiniz), batch normalization
# .. gibi metodlar var. Diğer bir yöntem de embedding_size boyutudur. Bu embbedding_size ı büyültüp küçülterek(50,100,300) deneyin mutlaka
# .. Diğer bir yöntem layer sayısını azaltmak. Zaten NLP modellerinde genelde layer sayısı 5 i geçmez yani tavsiye edilmez. Overfit e giderseniz layer sayısını azaltın
# .. Diğer bir yöntem learning rate ile oynamaktı(Altta ayarlayacağız)
# .. Diğer yöntemler early stop, batch_size la oynamak,  sample_weight= classes_weights ..(Bunların altta açıklaması var)
# class chat soru: o zaman biraz daha deneme yanilma seklinde olacak bu sayi hocam degil mi
# Johnson H: Aynen hocam
# class chat soru : Hocam burada hangi word embedding metodunu kullanacağını da belirtebiliyor muyuz?
# Johson H : Bu zaten DL modellerinde kullandığımız kendine has bir metoddur. Burada word2vec vs gibi bir şey kullanmıyoruz

# NOT: word embedding de  ilk değerler random olarak atanır

optimizer = Adam(learning_rate=0.008)
# optimizer: Minimum error u bulmamıza yarayan gradient descent algoritması
# optimizer : DL arka planda hangi gradient descent algoritmasını kullansın --> Genelde "Adam" kullanılır
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['Recall'])
# binary_crossentropy: Bunun(loss un) 0 olması demek neyi ifade ediyor?
# .. Bir yorumu %100 olasılıkla tahmin etmişsem log(1) = 0(loss)
# .. loss=0.001 Modeliniz o kadar iyi bir tahmin yapıyor ki 1 class ını ve 0 class ını neredeyse %100 iyi tahmin yapıyor(Örnek: log(0.996)=0.001 gibi... )
# HATIRLATMA: loss='binary_crossentropy' : multi class da loss='categorical_crossentropy' oluyordu

model.summary()
# https://stats.stackexchange.com/questions/328926/how-many-parameters-are-in-a-gated-recurrent-unit-gru-recurrent-neural-network
# Üsteki makalede alttaki formül var
# embedding_size * num_words ==> 50 * 15000 --->  embedding_11 (Embedding)    (None, 61, 50)            750000
# 3 x (n2 + nm + 2n) m= input, n= output
# (48*48 + 50*48 + 2*48)*3 ==> m = 50(embedding_size), n = 48(GRU dan kaç boyutlu çıkacak)
# (24*24 + 48*24 + 2*24)*3 ==> m = 48, n = 24
# (12*12 + 24*12 + 2*12)*3 ==> m = 24, n = 12
# 12*1 + 1

# Total params: 771,109 : Datam 771,109 katsayıyla işlem görmüş

early_stop = EarlyStopping(monitor="val_recall", mode="max", verbose=1, patience=1, restore_best_weights=True)
# early_stop : # overfit e gitmesini engellemek için early_stop u kullanıyorduk
# monitor="val_recall": Takip edeceği skor
# mode="max" : Takip edeceği skor maximum olması gerektiğini söylüyoruz
# mode a asla "auto" yazmayın. Eğer auto yazarsak trende bakar(düşme eğiliminde mi artma eğiliminde mi). val_recall: 0.9053, val_recall: 0.8464 görür ve
# .. der ki val_recall düşmesi gereken bir şey olarak algılar. Halbuki recall değerini arttırmak isteriz

# Johnson H: NLP de genelde 1-2 epoch da eğitimi bitirmemiz lazım çok çabuk overfit e gidiyor
# restore_best_weights=True : Bunu kullanmazsanız. Patience da eğitimi en son epochdaki katsayılar üzerinden metrikleri/skorları alır
# .. en iyi skoru aldığımız epoch a dönmez. en iyi skoru aldığımız epoch a dönmesi için restore_best_weights=True yapıyoruz

from sklearn.utils import class_weight
classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
pd.Series(classes_weights).unique()
# Datada imbalance lık durumu olduğu için  sample weight kullanacağız
# .. compute_sample_weigh: sample_weight i hesapla. Neye göre hesapla? --> y_train deki gözlemlere göre 
# .. nasıl yap bunu? --> class_weight='balanced' : hepsi dengeli olacak şekilde(y_train deki orana göre)
# 0 class ına 0.52, 1 clasına da 8.9 gibi bir ağırlık atamış (%95 e %5 di oran. %5 olan gözleme daha fazla ağırlık verdi(8.9))

model.fit(X_train, y_train, epochs=10, batch_size=256, sample_weight=classes_weights,
          validation_data=(X_test, y_test), callbacks=[early_stop])
# early stop kullandığımız için epochs=10 da olabilir epochs=100 de olabilir çok bir şey farketmez
# Overfitting varsa batch_size ile oynayacaktık.(2 nin katları şeklinde devam ediyordu)
# .. batch_size ın küçük olması eğitimimizin daha iyi olmasını sağlar (çoğunlukla). O yüzden overfitting varsa ilk olarak overfitting varsa
# .. batch size ı "büyüterek" kontrol edebiliriz
# sample_weight= classes_weights : Bu datanızdaki dengesiz olan sınıfın skorlarını iyileştirdiği gibi, overfit i de önler .
# !!! NOT: sample_weight de oranı kendisi belirliyor. class_weight de oranı kendimiz belirleyebiliyoruz
# validation_data=(X_test, y_test): validation datamıza test datasının kendisini verdik
# NOT: test setinin skorları, train seti skorlarından ilk epoch da daha yüksek çıkabilir. Alttaki linke bkz
# https://keras.io/getting_started/faq/#why-is-my-training-loss-much-higher-than-my-testing-loss
# Burada belirlenen(çıktıda gelen) "val_recall" 1 sınıfına ait skorlardır. O yüzden hedef label ı "1"  yapmayı unutmamalıyız

##### Model evaluation
# Bakalım overfitting var mı
model_loss = pd.DataFrame(model.history.history)
model_loss.head()

model_loss.plot()
# loss değerleri birbirine yaklaşırken recal değerleri birbirinden uzaklaşıyor
# !!! DL de eğer imbalance drumu varsa "loss" değerlerine kesinlikle bakmayınız
# !!! Çünkü %96 olumlu yorum, %4 ü olumsuz yorum. Olumluları %100 tahmin edip, olumsuzları örneğin  %65 gelecek ve ben bunların ortalamasını alsak
# .. % 94 vs gelir(ağırlıklandırılmış yapınca) ve loss düşer yani aslında olumsuzları çok yanlış tahmin yapıyor ama loss düşük.
# .. Böyle olunca loss a bakmak bizim yanlış yorum yapmamıza sebep olur.. Sonuç olarak NLP de data imbalance ise loss a göre yorum yapmayacağız
# !!! Eğer datamız çok küçükse DL de, hatanın local errorlara takılma olasılığı çok yüksektir(Yani global min e gelmez genelde)
# recalllar birbirinden uzaklaştığı için bu data overfite doğru gidiyor demek
# ML de accuracy ye bakmayın inbalance ise, DL de imbalance varsa loss a bakmayın
# !!! DL de cross validation yoktur. Çünkü her iterasyonda katsayılar farklı yerlerden rasgele atanır ve alınan errorlar farklı farklı olur(farklı local minimumlara takıldıkları için)
# .. batch_size larımız her epoch da train seti içinde karışır. Yani bir nevi cv yapılıyor 
# .. Evet cross validation yapılabiliyor bazı uygulamalarda ama katsayılar sabitlenip yapılıyor ama DL de katsayıların sabitlenmesi gibi
# .. bir durum yok. O yüzden DL de yaptığınız cross validation sağlıklı sonuçlar vermez

# class chat soru: hocam imbalanced datalarda loss'a bakmayalım dedik ama biz bu imbalance'lığı class_weights ile gidermiş olmadık mı ?
# Johnson H : Evet giderdik ama mesela Datamda 0 olumlu, 1 olumsuz olsun , birinde 100 tane olsun data, birinde 1000 tane olsun data. Datada dengeleme işlemi yapmazsam
# .. 100 tane çekersem aralarından 90 ı olumsuz, 10 u olumlu gelir ve sonra olumsuz sınıfa atayım gibi bir yorum yaparız. Yani fazla olan sınıfa yapılan
# !!! .. tahminler daha yüksek olur. Biz balance ı yaparak az olan sınıfa !!!daha fazla tahmin yaptırmış oluyoruz!!!. 
# .. Yani sonuç olarak az olan sınıfa yapılan tahmin sayısı arttırılıyor ancak bu loss değerlerini etkileyen bir durum değil. 
# .. Etkiliyor ama çok kayda değer bir düşüş olmuyor loss da

model.evaluate(X_test, y_test)
# Loss: [0.2822844684123993,
# recall :  0.9053016304969788]

model.evaluate(X_train, y_train)
# Loss: [0.25349363684654236,
# recall :  0.9552143216133118] # Bu 4. , 5. epoch da 1 e gelirdi(overfit e giderdi)
# Eğitimi devam ettirseydik bu recallar birbirinden uzaklaşırdı(0.9053016304969788, 0.9552143216133118)

y_pred = model.predict(X_test) >= 0.5
# model.predict(X_test): Önceden bu class döndürüyordu burada bana oran döndürü yani bu bana 1 classına ait olma olasılıklarını döndürür
# .. Bu oranlar 0.5 den büyükse True ya döndür küçük olanları False a döndür. O yüzden y_test ve y_pred i rahat bir şekilde karşılaştırabilirim
print(confusion_matrix(y_test, y_pred))
print("-------------------------------------------------------")
print(classification_report(y_test, y_pred))
# confusion_matrix in içine bunları verince true lar 1 olur, false lar 0 olur
# Aynı şekilde classification_report true lar 1 olur, false lar 0 olur

# 0 sınıfında  : 0       0.99      0.89      0.94     45965
# .. skorlarımızda kötüleşme var(Ağırlıklandırma yaptığımız için,  Yani tahmin etme olasılıkları düştüğü için ,
# .. 1 class ına yapılan tahmin sayısını arttır ki o skorları daha iyi yakalayabilelim dediğimiz için oldu

# 1       0.33      0.91      0.49      2735 :  # Bu skorlar(recall yüksek ama precision ın kötü olması) normal. Çünkü aşırı bir imbalance lık durum var burada
# .. ILerde BERT göreceğiz ve Bert in ne kadar güçlü olduğunu ve neden BERT modellerin tercih edildiğini göreceğiz
# Modelim 0.91 i yakalayabilmek için 2735 in yaklaşık 3 katı kadar(8100) tahminde bulunmuş. Evet ağırlıklandırma yapıyor
# .. ama böyle bir handicap var. Maliyet olarak geri dönecek bana. Bunlar olumlu mu olumsuz mu diye benim incelemem lazım(2735 i)
# .. çünkü bunların içine extradan 5400 civarı olumlu olan yorumlar kaçmış(precision 0.33 olduğu için bunları söyledik)
# O yüzden recall u yüksek yapıyoruz ama biz precision ı da makul ölçüde yükseltmeliyiz. Yani modelimiz iyi tahminler yapmalı

y_train_pred = model.predict(X_train) >= 0.5
print(confusion_matrix(y_train, y_train_pred))
print("-------------------------------------------------------")
print(classification_report(y_train, y_train_pred))

y_pred_proba = model.predict(X_test) #!!! predict burada 1 label ının probasını döndürür.(predict_proba 1 ve 0 sınıflarının probalarını veriyordu)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
plt.plot([1, 0], [0, 1], 'k--')
plt.plot(recall, precision)
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('precision recall curve')
plt.show()

average_precision_score(y_test, y_pred_proba) # modelin genel performansı %73 benim modelim negatif sınıfları, pozitif sınıflardan ayırmada
# .. %73 oranında başarılıymış

# NOT: Overfit i engellemek için yapılabilecekler
# 1.Dropout ve içindeki oran
# 2.Batch normalization
# 3.Embbedding_size
# 4.unit sayısı
# 5.learning rate
# 6.data dengesizse "balance" durumları (class_weight, sample_weight)
# 7.restore_best_weight =True yapmayı unutmuyoruz
# 8.batch_size büyülterek, küçülterek
# 9.test_size 'ı düşürmek
# 10.Bunların hiç biri overfiti engellemiyorsa epoch=1 yapın, tek epoch la işi bitirin

# %% NLP-6
# Scores without sample_weight
"""
# Burada overfit var gördüğümüz gibi
Test set

            precision    recall  f1-score   support

       0       0.98      0.99      0.98     45965
       1       0.72      0.63      0.67      2735

accuracy                           0.97     48700
Train set

          precision    recall  f1-score   support

       0       0.99      0.99      0.99    183856
       1       0.87      0.80      0.84     10941

accuracy                           0.98    194797
"""

##### Model Saving
# model.save('review_hepsiburada.h5') # Bunu indirip sonra drive a yükleyin(böyle hata almazsınız)

# Loading
#from tensorflow.keras.models import load_model
#model_review = load_model('/content/drive/MyDrive/review_hepsiburada.h5')

# class chat soru: NLP'de model başarısı olarak yüzde kaç iyidir? Dataya göre yine değişir mi?
# Johnson H : Değişir hocam. Her zaman 80 li 90 lı skorlar almayı beklemeyin
# .. BERT model her türlü şeyi yakalıyor diyoruz ama farklı bir cümle gelince model şaşırabilir o cümle olumlu mu olumsuz mu
# .. Örnek cümle: Anyayı Konyayı görürsünüz telefonu alınca... gibi bir cümle görmezse eğer model, tahmin yanlış olabilir(BERT de bile)

##### Prediction
review1 = "çok beğendim herkese tavsiye ederim"
review2 = "süper ürün"
review3 = "büyük bir hayal kırıklığı yaşadım bu ürünü bu markaya yakıştıramadım"
review4 = "kelimelerle tarif edilemez"
review5 = "tasarımı harika ancak kargo çok geç geldi ve ürün açılmıştı hayal kırıklığı gerçekten"
review6 = "hiç resimde gösterildiği gibi değil"
review7 = "kötü yorumlar gözümü korkutmuştu ancak hiçbir sorun yaşamadım."
review8 = "hiç bu kadar kötü bir satıcıya denk gelmemiştim ürün siparişimi iptal ediyorum"
review9 = "tam bir fiyat performans ürünü"
review10 = "beklediğim gibi çıkmadı"
review11 = "on numara beş yıldız"
review12 = "üründe sıkıntı var"
review13 = "televizyonun görüntü kalitesi çok kötü, dün akşam evde arkadaşlarla toplandık. yedik, içtik, sohbet ettik. Sonra televizyonda Türkiye - İngiltere maçını seyrettik."
review14 = '1 gün gibi kısa bir sürede elime geçti. Ve bu fıyata süper bir ürün tavsiye ederim. Lakin eli büyük olan kişiler daha büyük modelini tercih edebilirler ortaboy ürün.Teşekkürler '
review15 = 'telefon kesinlikle bu parayı hak etmiyor. yeni hiç bir özelliği yok. ancak küçük ekran sevenler için iyi bir telefon'
review16 = 'müthiş bir telefon ama bu parayı hak etmiyor'
reviews = [review1, review2, review3, review4, review5, review6, review7, review8, review9, review10, review11, review12, review13, review14, review15, review16]
# Modelleri eğitime verirken tokenizer kullanmıştık. 

tokens = tokenizer.texts_to_sequences(reviews) # Corpusumda geçen ilk 15000 tokene göre dönüşüm yaptırıyor
# Üstte geçen cümlelerdeki kelimeler(tokenler) ilk 15000 içinde değilse onlar ignore edilecek
# Johnson H: Ben tüm data ile de sonuçlar aldım ama skorlarda çok bir değişiklik olmadı. İsterseniz deneyebilirsiniz
# tokenizer.texts_to_sequences(reviews) : Yorumları en sık kullanılana ilk 15000 tokeni dikkate alarak nümeriğe dönüştür

tokens
# [1, 146, 70, 9, 10] : Üstteki 1. cümle (review1)
# [73, 4] : Üstteki 2. cümle (review2)
# .. Örneğin review4 de 1 token ignore edilmiş

tokens_pad = pad_sequences(tokens, maxlen=max_tokens)
tokens_pad.shape
# 16 yorumun her birini 61 e sabitledi
# maxlen = None yazılırsa en uzun yoruma göre sabitler(bütün yorumları 298 e sabitler) 
# .. Ancak maxlen = 61 yazılırsa yani integer girdiğinizde hata alırsınız. 61 i bir değişkene eşitleyip öyle vermeliyiz maxlen e
prob = model.predict(tokens_pad)
prob
# modelim eğitildi artık ve her bir tokenin word embeddingleri belirlendi. 
# !!! NOT: Burada yapılan işlem: tokens_pad ler içerisine ne olarak verildi? Üstte "tokens" kodunda gelen sayılara göre
# .. Ama biz eğitimi word embeddingler üzerinden yaptık fakat burada hiç bir word embedding e dönüşümü yok nasıl tahmin yapacağız?
# .. Burada modelim artık eğitildi zaten ve ilk 15000 tokene göre word embedding leri belirlendir
# .. O yüzden mesela 146 ya karşılık gelen word embedding ne ise predict aşamasında öncelikle
# ..  word embedding değerleri atanıyor tokenlere ondan sonra predict yapılıyor
# Tahminleri yaptıktan sonra oranları döndürmüş


pred = (model.predict(tokens_pad) >0.5).astype("int")
pred
# Bu oranlara göre, bunları 0.5 den büyük olanlar TRUE olacak. Küçükler False
# Truelar -->1 , false lar 0 oldu

pd.set_option('display.max_colwidth', None)
my_dict = {"Review":reviews, "prob":[i[0] for i in prob], "classes":[i[0] for i in pred]}
pd.DataFrame(my_dict)
# Yorumlarım(Review) tek boyutlu iken problarım ve class larım 2 boyutlu. Ya hepsi tek boyutlu olacak ya hepsi 2 boyutlu 
# Hepsi aynı boyutlu olsun diye ""[i[0] for i in prob]"  ve "[i[0] for i in pred]" yapıyoruz ve tek
# .. boyuta dönmüş oluyor. ( i[0] : listenin içindeki değeri almak için yazdık. Bunu üstte array içinde yapmıştık)
# Bunuda en son df e dönüştürdürünce;
# Yorumları olasılıkları ve class predictionlarını görüyoruz
# pd.set_option('display.max_colwidth', None): df i n hücre genişliğini en uzun yoruma göre ayarla(O yüzden Review sütunu
# .. en geniş haline gelmiş ancak 12. ve 13. yorumda 2. satıra geçme olmuş. 2. satırda da son kelime satırın son kısmına 
# .. gelecek şekilde yazılmış/ayarlanmış)
# NOT: Burada problar 1 class ına ait olma olasılığı
# NOT: Yorumlara bakarsak hoca bir kaç şey söyledi ama genel anlamda 4. indexteki yoruma bakarsak
# .. model "hayal kırıklığı" kelimesine ağırlık verdiği için orada "tasarımı harika" görse bile bunu dikkate almamış ve olumsuz vermiş
# .. diğer kafa karıştıran yorumlar için hep bir kelimenin ya da yapının diğer kelime ya da yapılara ağırlık vermesinden dolayı
# .. tahminleri çok doğru ya da çok yanlış yaptığını söyleyebiliriz
# .. 6. indexteki yorumu neden bilemedi?(kötü yorumlar gözü....) .. Burada olumlu yorumları yakalayamamışsa bu bizim verdiğimiz
# .. ağırlıktan kaynaklandığını düşünün
# .. 14. indexte "bu parayı hak etmiyor" u görünce direk olumsuz demiş. Bunu alttaki kodda kontrol edelim

# class chat not : 1 eklemeyince hata veriyor bütün tokenleri kullandığımız durumda böyle bi çözüm buldum
# .. https://keras.io/api/layers/core_layers/embedding/
# Johnson H: num_words ataması yaparken 1 artırarak kaydederseniz sorun olmaz. imput_dim = num_wordse eşit olmak zorunda

count=0
for i, j in enumerate(X):
  if "hak etmiyor" in j.lower() and y[i]==1:
    count+=1
count
# "hak etmiyor" olumlu yorumlarda 1 defa, olumsuz yorumlarda 31 defa geçiyor. O yüzden "hak etmiyor"
# .. görünce olumsuz demiş

# DL de her zaman her çalıştırdığınızda skorları kaydedin çünkü yoksa aynı yüksek skorları bir daha yakalayamayabilirsiniz

########## BERT(Bidirectional Encoder Representations from Transformers)
# 2018 de GOOGLE tarafından tanıttı. Transformers lar 2017
"""
                            BERT
                        Transformer
                          Attention
                Encoder-Decoder  Bi-LSTM
                         RNN     LSTM
"""
# Altyapısında transformers ları kullanıyor
# NLP nin DL ile kullanılması 2013 lere(RNN) lere dayanıyor.
# LSTM i falan time series lerde kullanılıyor 1997 olmasına rağmen. 2013 te bunu NLP ye uygulamışlar
# Encoder-Decoder: LSTM ve GRU nun birlikte kullanıldığı. Translation, question -answers, sonraki cümleyi tahmin etme, digital assistants vs
# .. İçerisinde 1 den fazla LSTM ve GRU modelleri kullanılarak bu işlemleri yapar(Translation vs). Ancak bunlara gerek kalmadı
# .. Çünkü Transformers lar altyapısında encoder-decoder ları kullanıyor.
# Bi-LSTM : LSTM in çift taraflı çalışması
# Attention: Transformersların altyapısında kullanılıyor. Textlerde geçen tokenler arasındaki anlamsal ilişkileri çok daha güçlü bir şekilde kuruyor
# .. attention yapıları
# Transformers : Içinde attention mekanizmaları ve yeni nesil encoder-decoder yapıları var. Encoder a input u veriyoruz, decoder a output u veriyoruz
# BERT : Alt yapısında transformers yapısı kullanıyor 12 tane transformers yapısı kullanıyor. 
# Neden encoder-decoder yapılarını anlatmıyoruz. Çünkü genelde zaten transformers ların 
# .. oluşturduğu modeller tercih ediliyor çünkü sonuçları çok iyi veriyor. BERT bunlardan birisi

# Transformers lar
    # Encoder  : Modele input u verdiğim kısım     (input: how are you?)
    # Decoder : Modele output u verdiğimiz kısım (output: İyiyim)

# Bert modellerinin içerisinde kullanılan transformers yapısı, "transformers" lardan biraz farklı 
# .. BERT teki transformers lar decoder yerine de encoder kullanıyor.
#  encoder a input olarak "what are you doing" diyoruz ve outputda karşılığı geliyor
# ... "was macsht du geralde". inputu ve ouputu birlikte verip 2 sini de aynı anda öğrenmesini istiyoruz

# Positional embedding
#  Bir modele tokeni verirken embedding şeklinde vereceğim. Bert modelin hafızasında word embedding ler mevcut
# .. yani "what time is it" i modele verirken modelde "what" a ait word embedding belli, "time" a ait belli
# .. vs vs.. ve biz bu tokenlerin word embedding bilgisini input olarak veriyoruz. Buna karşılık gelen "ten o'clock"
# .. un word embedding lerini de output a vereceğiz
# Önceden modeller(RNN, word2vec veya glove, LSTM, GRU) seri olarak çalışıryorlardı, yani What a yapacağı bitmeden time a işlem yapmıyor, 
# ..time a yapacağı işlem bitmeden "is" e işlem yapmıyordu vs vs.
# BERT, paralel olarak çalışır yani bu texti bütün olarak alır. "what time is it" i .
# .. Bu sefer de şöyle bir handicap ım oluyor. Bu tokenlerin(what time is it?) cümlede
# .. kaçıncı sırada olduğunu bilmiyor. Sıralamasını biz positional encoding le bildiriyorum modele
# .. peki modele bütün olarak vermenin ne gibi avantajları var

# Örn: Öğretmen öğrenciye ilk öğretim yılında okuldu çok başarılı projeler yaptı.
# Bunlara klasik LSTM ve GRU yaptığımızda sırayla(seri) çalışacak. Ancak LSTM ve GRU anlamsal ilişkileri
# .. birbirine yakın olan tokenlerle daha iyi kuruyordu uzakta kalanlara göre
# BERT de bir bütün olarak beslediğimde "attention" mekanizması sayesinde, modelim sadece birbirine yakın olan tokenlerle değil
# .. sanki cümlenin öğelerine sorular soruyor gibi yakalıyor. Anlamsal ilişkileri cümlenin zarfına, sıfatına bakarak kuruyor
# .. nerede yaptı "okulda".. kim yaptı "öğretmen" vs vs. Peki bunu nasıl biliyor? Johnson H: Bunu anlatırsak ders bitmez
# .. Özet olarak burada unutma problemi yok. Text i bir bütün olarak görüyor burada(LSTM ve GRU da böyle değil)
# Multi-head attention: Anlamsal ilişkileri "bir çok farklı" şekilde yakalayan mekanizma

# Attention mekanizması "self attention" olarak adlandırılır. 
# Self-attention: 2 token arasındaki anlamsal ilişkinin tek bir yön ile ele alınması(). Öğretmen ve öğrencinin sadece "okul" üzerinden kurulması self attention dır
# Multi-head attention: sadece "okul" ilişkinin yerine bir çok yön ile kurulması(öğrenci, okul, not)

# Attention nlamsal ilişkileri nasıl kuruyor? Query, key ve value vektörleri üzerine kuruyor
# Query: Tokenin diğer tokenlerle olan ilişkisini öğrenmek istiyorum (Öğretmen tokeni diğer hangi tokenlerle ilişkili)
# Key  : Tokenin ilişkili olduğu diğer tokenler(öğrenci, okul, not, ...)
# Value: Tokeni hangi text içinde kullanmışsanız o text in içeriğine bakarak öğretmen ile
# ..  ile en anlamlı tokenler hangisi ise o tokeni belirliyor.
# ... Sonra context in içerisindeki yapıya bakarak "öğretmen" ile en ilişkili "öğrenci"(bu value dur) diyor
# Model öğretmen ile en yüksek anlamsal ilişki hangileri ise önce onlara yoğunlaşır.(LSTM ve GRU bütün tokenlere yoğunlaşıyordu)
# .. Attention mekanizması bu şekilde çalışması belli başlı tokenlere yoğunlaşıp sadece onlar üzerinden anlamsal ilişkiyi bulması
# .. çalışma maliyeti azalmış olarak geri dönüyor

# Örnek1: The animal didn't cross the "street" because it was too long
# The animal didn't cross the street because "it" was too long

# Örnek2: The "animal" didn't cross the street because it was too tired
# The animal didn't cross the street because "it" was too tired

# Burada "it" ile hangi tokenler daha anlamsal ilişkili onları bulmaya çalışıyorum
# Modelim bir çok tokenle eğitildi ve aldığı eğitim sonrasında modelim neyi öğrendi?
# Birinci cümlede "long" olan ancak bir "street" dir(cansız bir şeydir) der ve modelim "long" dan dolayı 
# .. "it" ile "street" arasındaki ilişkiyi kuruyor
# Ikinci cümlede "tired" yani canlı bir şey yorgun olur, cansız bir şey yorgun olamaz der(Buna aldığı eğitimler sonrasında karar verdi)
# Yani cümleyi bütün olarak gördüğü için anlamsal ilişkiler daha iyi kuruldu

# Multi-head attention: Bundan bahsetmiştik. Ne yapıyordu. Ilişkinin yerine bir çok yön ile kurulması
# Bütün tokenlere hakim olduğu için model öğretmen-öğrenci, öğretmen-not verme, öğretmen-okul vs yi görüyor
# .. ve bu tokenlere odaklanıyor "öğretmen" için

# LSTM ve GRU uzun cümlelerde iyi sonuçlar verse de o da bir yerde yetersiz kalıyor. Çünkü keywordlerim çok olursa LSTM ve GRU da sıkıntı yaşayabiliyor
# En iyi mekanizmalar transformerslardır(NOT: 512 tokenden fazlasını alamıyor şu an. Eğer daha çok geliştirilirse
# .. unutma probleminin ortadan kaldırılabileceği söyleniyor. 9 yılda NLP de neler neler olmuş. Bu da neden olmasın)

# Örneğin model "How are you" görünce
# "You -- I " ile anlamsal ilişki kurmuş
# "are -- am" ile anlamsal ilişki kurmuş
# "how" -- fine ile anlamsal ilişki kurmuş
# "You" geldiğinde "I" ı kastettiğini anlayıp "I" ı getiriyor önümüze model
# Modelim anlamsal ilişkileri o kadar iyi yakalamış ki;

# Peki model sadece bütün tokenleri aynı anda text te gördüğünden dolayı mı bu kadar başarılı? Hayır. Ayrıca;
# Diyelim model eğitilirken bir text im var diyelim. text 200 tokenden oluşsun
# !!! Her bir cümlede örneğin 200 tokenin %15 ini alır. Bu 30 tokenin(%15 in) %80 ini maskeler ve
# .. % 10 unu olduğu gibi bırakır ve  %10 unu da başka bir tokenle değiştirir. BU EĞİTİM AŞAMASI. YANİ EĞİTİM DEVAM EDİYOR
# .. Eğitim devam ederken arka planda bunların kontrolünü yaparak "loss" u da sürekli günceller.
# .. model bu maskelediklerimi doğru tahmin edecek mi ona bakıyor, bir tokenle başka bir tokenin değiştirildiğini
# .. anlayacak mı anlamayacak mı ona bakıyor BERT (!!! DAHA EĞİTİM DEVAM EDERKEN). Yani modeli sürekli kontrol ediyor
# .. ve değiştirmesi gereken bir şey varsa onu ayarlıyor BERT
# !!! Eski modeller için mesela öğrenciyi sınava hazırlarken sürekli sorular çözdürüyorum
# .. hiç bir deneme yapmamış ve sadece üniversite sınavına giriyor. BERT böyle yapmıyor. Eğitim sırasında 100 tanesini alıyor
# .. deniyor. Sonra başka alıyor deniyor. Yani eğitim esnasında arkada yapması gereken başka bir şey varsa bir yönden onları ayarlıyor arkada
# Özetle; Eğitim devam ederken modeli bir yandan denemelere/teste tabi tutuyor(yani eğitimin bitmesini beklemiyor) bu yüzden sonuçları daha iyi geliyor diğer modellere göre
# .. sürekli bulduğu loss değerindeki azalış ve artışa göre tekrar denemeler yapıyor

# Transformers ların çıkışı NLP için bir devrimdir(2017). BERT 2018
# Bert in en temel modelleri;
    # BERT Base  : 12 layer(transformers bloğu) kullanır. Her transformers bloğunda 6 encoder 6 decoder kullanır. 
        # .. 12 attention heads(bunlar anlamsal ilişkileri kuruyordu). 110 milyon parametre ile işlem görmüş(eğitilmiş) modelimiz
    # BERT Large : 24 layer kullanır.Her transformers bloğunda 6 encoder 6 decoder kullanır. 
        # .. 16 attention heads(bunlar anlamsal ilişkileri kuruyordu) 340 milyon parametre ile işlem görmüş(eğitilmiş) modelimiz
# NOT: Generative Pre-trained Transformer 3,  175 milyar parametre ile eğitilmiş. Bunun eğitimi 40 gün sürmüş(Bir de o kadar kuvvetli makinalarla)
# NOT: Bunların üzerine yeni modeller geliştirildi "bert turk" gibi. Önce sadece english vardı ama şimdi her dil var neredeyse
# Bert in eğitimi 4 gün sürmüş. 800 milyon farklı kelime ile ve wikipediadaki 2.5 milyar kelime ile eğitilmiş
# Mask language modelling(MLM) and next sentence prediction(NSP) yaparak eğitiliyor(Yukarda maskeleme ile ilgili kısımdan bahsetmiştik)

# Örnek cümle:  [CLS] I love [MAKS] drink [MASK] [SEP] and I hate tea
# CLS : Bir cümlenin başlangıcını gösterir
# SEP: Bir cümlenin bittiği yeri gösterir
# BERT bunları mutlaka ister. Bunu yapmazsanız BERT model hata verir
# MASK : Maskelediği token. Bakalım eğitimden sonra bu MASK ları doğru tahmin edecek mi model?
# .. eğitim devam ederken, 2 tane mask var burada ve tahmin yerinde 3  tane MLM classifier var. 
# .. üçüncüsü ya bir token aynı tahmin edilmiş ya da değiştirilmiş o yüzden 3 tane
# Aynı anda model bu işlemleri yaparken bir sonraki cümleyi de tahmin etmeye(NSP) çalışıyor(SEP den sonraki kısım)

# BERT in kullanıldığı alanlar
    # Machine Translation
    # Question Answering
    # Text and Token classification

# BERT in de sınırları var
# Eğitim için kullanacağı max token sayısı 512 dir.. Başka bir deyişle; Bir satırda eğitim için verilebilecek max token sayısı 512 dir.
# Kullandığımız word embedding boyutları 768(bert-base) , 1024(bert-large)
# DL modellerinde, RNN modellerinde olduğu gibi bütün tokenler modelle sabit uzunlukta olmalı.(Göreceğiz)

# Bert modellerin başarılı olmasının sebeplerinden 3. cüsü; kullandığı tokenization işlemi. Bu wordpiece tokenization işlemi
# wordpiece: Bir cümle içerisindeki kelimeleri tokenlere ayırırken köklerini ve eklerini ayırarak o şekilde hafızasında tutar.
# .. Her bir kelime ve ek için bir word embedding döndürür
# Strawberry: çilek , straw = x, berry =y ise bunları ağırlandırıp yeni bir word embedding(z) elde ediyor
# Önceki sistemlerde araba: a, arabacı:b, sigorta : c, sigortacı: d, şeklinde tutuyordu. hafızamızda çok fazla token tutma zorunda kalıyoruz
# .. ancak wordpiece de , arabacı geçersek araba ya ait "a" ile cı ya ait "b" yi (örneğin b olsun dedik. Yukardaki b ile bağlantısı yok) 
# .. alıp ağırlandırılmış şekilde birleştirecek
# .. Yani "araba" ve "arabacı","sigorta", "sigortacı" yı ayrı ayrı tutmayacak hafızada. İhtiyacı olan "a" yı, "b" yi "c" yi vs alacak.
# .. BERT hafızasında 30522 tane token tutuyor(Az aslında). Çünkü hafızada ekleri ve kökleri ayrı ayrı tuttuğu için her kelime için bir word embedding döndürür
# .. Modele ne verirseniz verin mutlaka bir word embedding döndürür BERT

"""
Input            : [CLS]  my      dog    is  cute    [SEP]     he    likes   play    ##ing    [SEP]
Token Embeddings : E[CLS] Emy   Edog    Eis  Ecute    E[SEP]   Ehe  Elikes   Eplay    E##ing    E[SEP]
Segment Embeddings: EA    EA     EA     EA     EA      EA       EB     EB      EB       EB       EB
Position Embeddings:E0    E1     E2    E3      E4      E5       E6      E7      E8      E9      E10  

# Normalde biz DL e bir text i verdiğimiz zaten DL nin istediği bir tane vektördü 

# my: bert modelinde 20. token
# dog : bert modelinde 300. token vs vs
# NOT: CLS : 2, SEP:3 tür. Bunlar sabittir
# Verilen bir texti BERT de hafızasındaki 30500 tokeni kullanarak sayısallaştırılıyor
# BERT bizden 3 vektör ister. Token Embeddings, Segment Embeddings ve Position Embedding dir
# .. BERT modelindeki en zor kısım burası. Notebook da daha iyi anlaşılacak

# Token Embeddings(input_ids) : Text in nümerik forma dönüştürülmesi(DL ile aynı)(Hafızada tokenim kaçıncı sırada)
# Segment Embeddings(token_type_ids): Classification ve sentiment analizde kullanmayacağız çünkü classification da vs 2 farklı cümle olmayacak
# .. eğer 2 cümle vermek zorunda kalsaydım inputlarımı(EA) ve outputları(EB) göstermek zorundaydım.
# .. Bu vektörü ama "translation", "question-answers", "diğer cümleyle benzer" vs gibi bir uygulama şey olsaydı
# .. EA lar benim inputlarım olacak, EB lerde benim outputlarım olacak. Inputlar 0, outputlar 1 . Model bunları bilmezse
# .. inputları hangi encoder a göndereceğini , outputlara işlem yapacağım encoder göndereceğini model bilemez
# .. Bunu kullanmayacağız biz çünkü sadece classification ve sentiment yapacağımız için
# Position Embeddings(attention_mask) : Modelimin gerçekteki uzunluğunu bildiriyor. Her tokene 1 numarası atar ve  o
# .. 1 leri sayar ve sadece 1 lere işlem yapacak. Hangi tokenlere işlem yapacağını bilecek kalanı 0 ile dolduracak(token olmayan yerlere)(padding ile)
# .. Örneğin 10 token var diyelim.Bunların hepsi 1 ile işaretli. Modele Sen sadece 1 lerle işaretli olanlara işlem yap, 0 lara işlem yapma diyoruz
# .. Her tokene "1" numarasını veriyoruz ve "1" rakamı verdiklerim arasında anlamsal ilişkiler kursun diyoruz burada

# Burada token ve position embedding kullanacağız biz
"""
# (NLP-6 - 2:56:00): Üsttekileri şekillerle açıklıyor

# class chat soru: sormak istediğim dersin başında tokenleri paralel verdiğimiz için sırasını bilmiyor. 
# .. Sırayı vermemiz lazım demiştiniz onu sormuştum hocam bu sırayı nasıl veriyoruz?
# Johnson H: Position encoding le bert modelime hangi sırada olduğunu söylüyor
# class chat soru: positional encoding ile positional embedding kavramları karıştı Hocam
# Johnson H: positional encoding: bütün tokenleri aynı anda tokenleri verdiğimiz için positional encoding le tokenlerin hangi sırada olduğunu anlıyor
# .. positional embedding : Hangi tokenlere işlem yapacağım(1 ile işaretlilere(gerçekten var olan tokenlere) işlem yapacak, 0 la işaretlilere(padding yapılanlara) işlem yapmayacak)
# .. "what time is it ?" # Positional encoding vasıtasıyla "what":1. sırada, time:2. sırada is:3. sırada it:4. sırada ?: 5. sırada ... ya bakar. 
# .. positional embedding "1 1 1 1 1"  ile doldurma işlemi  i de kendisi yapıyor(NOT: Bunu da kendisi yapıyor. parametre olarak vermiyoruz)
# .. Sonuç olarak : 1-2-3-4-5 işlemine "encoding" .. 1-1-1-1-1-0-0-0 işlemine de "embedding" 

"""
Johnson Hoca:
Merhaba arkadaşlar. Dün dersimizde kafa karışıklığına neden olan positional encoding ve positional embeddings konularına ilave açıklama getirmek istiyorum.
Klasik transformer mimarisinde kullanılan positional encoding cümledeki tokenlerin sıralama bilgisini modele verir. Buradaki sıralama bilgisi tokenlerin word embedinglerinin içine işlenir.
Positional embeddings ise  modelin işlem yapacağı tokenleri belirlemek için kullanılır ancak altyapısında transformers kullanan bazı modeller positional encoding'in yaptığı gibi positional embeddingslerdeki 1 numaralarının içine tokenlerin position bilgilerine de ilave eder. İlave etmesinin sebebi ise developerların bu şekilde modellerin daha iyi öğreneceğini düşünmelerinden kaynaklanıyor.
Okuduğunuz kaynaklarda genellikle positional embeddingin işlem yapılacak tokenleri belirleme kullanıldığı yazmaz ve sadece tokenlerin sıralamasını verir bilgisi yer alır. Daha doğrusu positional encoding ve positional embeddings farklarına detaylı değinilmediğinden kafa karışıklığı olabiliyor.
BERT modellerinde hem positional encodings hemde positional embeddings kullanılır.
"""
#%% LAB-1



#%% NLP-7
# Bu gün BERT modellere fine tuning yapılmasını öğrendikten sonra aklınıza gelecebilecek her türlü classification ve sentiment analizinde bunları 
# .. kullanabilirsiniz
# NLP nin diğer task lerinin(translation, sonraki cümleyi tahmin vs vs) kendi özel dataları var ama bunları bulmanız çok zor. 
# .. Ancak classification da  her yerden farklı farklı data bulabilirsiniz önemli olan şey buna fine tuning uygulayabilmek.
# Ancak elektronik cihazlarla eğitilmiş bir model ile araba ile ilgili yapılan yorumları tahmin edemezsiniz. 
# Yani önemli olan datayı alıp sizin Bert modele sokup fine tuning yapmanız lazım. Bu gün bunu göreceğiz

# Alttaki kodlar tensorflow un kendi sayfasında verdiği hazır kodlar. NOT: tensorflow haricinde TPU kullanamazsınız
"""
Johnson H not:
notebook ayarlarından TPU'yu seçip kaydedin ve ilk 2 hücreyi çalıştırsın. 
Boşta TPU çekirdeği yok uyarısı alan arkadaşlar 2-3 dakikaya bir bu hücreleri tekrar çalıştırsınlar.
Uyarı alanlara genelde en geç 5-10 dakika içinde çekirdekler tahsis ediliyor.  O yüzden panik yapmayın
"""
# Note that the `tpu` argument is for Colab-only
import tensorflow as tf
import os
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))
# Bu boşta çekirdek varsa çekirdekleri tahsis edecek. all-device diyor, 8 tane cihazın ismini yazdırıyor

strategy = tf.distribute.TPUStrategy(resolver)  # Bu da hazır kod(tensorflow sayfasından)
# TPU çekirdeklerinin hızından faydalanabilmek için aşağıda bir fonksiyon altında yaptığımız bu modeli tanımlamamız lazım
# .. eğer o fonksiyon altında modelimizi tanımlamazsak TPU çekirdeklerinin hızından faydalanamayız. O yüzden
# .. "strategy" denilen bir değişkene atadık

# from google.colab import drive
# drive.mount('/content/drive')

# !pip install transformers

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

df = pd.read_csv('/content/drive/MyDrive/hepsiburada.csv', encoding="utf-8")
df.head()
df.Rating = df.Rating.map({1:0,0:1})
df.Rating.value_counts()                # 229821 : olumlu, 13676 :olumsuz
df.Rating.value_counts(normalize=True)  # 0.943835: olumlu   0.056165: olumsuz

##### TRAIN-TEST SPLIT
X = df['Review'].values # X ve Y yi array e dönüştürdük. Çok büyük datalarda ML de arrayler daha hızlı çalışır(DL de zaten array olmak zorundaydı. Yoksa çalışmaz)
y = df['Rating'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# Önce bir örnek yapalım
##### Tokenization Example For BERT
# https://huggingface.co/dbmdz/bert-base-turkish-uncased
# Linkte bu tokenizer ın nasıl kullanılacağına dair bilgileri vermişler. Bunları kopyalayıp yapıştırdık alta
# case   : büyük küçük harfe duyarlı şekilde token işlemleri yapar.(Translationda case kullanmak gerekir örneğin virgül, nokta kullanınca anlam değişir. Büyük harf ona göre değişir vs)
# uncase : Hepsini küçük harfe dönüştürüp token işlemleri yapar(Classification da case e ihtiyaç duymuyoruz o yüzden bunu kullanıyoruz)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")

text = "Şentepeli Şükrü abi çok güzel bir insan?😊"
tokenizer.tokenize(text)
# Bert emojileride anlayabiliyor. Örneğin gülen yüzün olumlu olduğunu anlıyor
# Örneğin I (kalp işareti) you olsun. Bu normalde I love you anlamında ancak önceki modellerde bu kalp işareti algılanmayacak ve elimizde I ve you kalınca
# .. bunlarda stopword olduğu için elimizde bir şey kalmayacak(ML de). Eğer DL de olsaydık da elimizde I ve you kalırdı buradan bir analiz çıkaramazdı model
# .. Ancak bert bunları anlayabiliyor
# Çıktıda "##epe": word piece tokenizer(eklerine ve köklerine ayırıyordu) yapılmış(eklerine ayırmış)
# .. 12 tane ayrı ayrı tokene ayırmış. Normalde bildiğimiz tokenleştirme yapsaydık "?" ve "emoji" dahil 9 tane token verecekti
# NOT: Burada türkçe karakterler yok ama modelimiz bunları türkçe olarak anlamsal karşılıklarını buluyor bert türk

sentence = "Şentepeli Şükrü abi çok güzel bir insan?😊"   
tokens = tokenizer.encode(sentence, add_special_tokens=True) # Sayısal forma dönüştürme(DL de yaptığımız gibi)
print(tokens)
print(len(tokens)) # 14. word piece e göre kaç tokenden oluştuğu bilgisini verdi(CLS ve SEP i de ekledik o yüzden 14 oldu)
# "sent" 15955 inci tokene denk geliyormuş," ##epe" 11679 inci tokene denk geliyormuş vs vs
# add_special_tokens=True: Bert modellerinde 2 özel tokenimiz vardı CLS[2] , SEP[3]

# Bütün yorumlarımın kaç tokenden oluştuğunu öğreneyim ve bütün yorumları aynı boyuta sabitleyeceğiz
##### Fixing token counts of all documents
# For every sentence...
max_token = []
for sent in X:
    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sent, add_special_tokens=True) # add_special_tokens=True: CLS[2] , SEP[3]
    max_token.append(len(input_ids))
print('Max sentence length: ', max(max_token))
# Her bir cümleyi alıp(for sent in X), sonra encode ile sayısallaştıracağız ve başına ve sonuna özel tokenlerimizi ilave edeceğiz(add_special_tokens=True)
# .. sonra içerisindeki tokenleri sayıp(len(input_ids)) , max_token listesinin içine ekleyeceğiz(append)
# En sonda da print ile Max_token içinde her bir yorumun kaç tokenden oluştuğu bilgisi yer aldıktan sonra en uzun yorumun kaç tokenden oluştuğunu göreceğiz
# En uzun yorum 546 imiş
# NOT: Transformers lare 512 tokenden daha uzun olan cümleye işlem yapmaz. Hata alırsınız. O yüzden ne yapmamız lazım
# .. daha az bir token sayısına sabitlememiz gerekli.

"""
list = [5, 10, 8, 9, 12, 15, 4]
print(np.array(list) <= 10)
print(sum(np.array(list) <= 10)/len(list))
# Yüzde 95 inde bilgi kaybı olmasın demiştik önceden. Yine bu mantık anlatılıyor burada
"""
np.array(max_token).mean() # Corpusumdaki yorumlar ortalama 39 tokenden oluşuyor
sum(np.array(max_token) < 160) / len(max_token) # Johnson H: Burada ben 160(boyut) a karar kıldım # %98 e denk geliyor
# Bert modele sayısallaştırdığımız verileri kaç boyutta vereceğimize karar kıldık(160)

##### Transformation Vectors to Matrices
# Her bir text imiz bir vektöre dönüştü ancak. Bert model 3 tane ayrı ayrı vektör ister.

# Burada encode(yukarıda encode kullanmıştık) yerine encode_plus kullanıyoruz
sentence = "Şentepeli Şükrü abi çok güzel bir insan?😊"
tokens = tokenizer.encode_plus(sentence, add_special_tokens=True)
print(tokens)
# input_ids: tokenlerime bir kimlik numarası veriyor
# token_type_ids: translation veya question & answers gibi modellerde modele verilen textlerin hangisi input, 
# .. hangisi output olduğunu belirliyordu. inputlar 0, outputlar 1 oluyordu. Bu classification da ve sentiment 
# .. analizde kullanmayacağız o yüzden hepsi 0 gelmiş zaten.
# attention_mask: Position embedding. Bunların gerçek boyutlarını(uzunluğunu) öğrenme hem de pozition bilgiler.
# .. hepsine 1 numarası verip "1" leri sayıyordu
# !!!Bert modelin benden istediği 3 ayrı vektörü elde ediyorum. Bunun için "encode_plus" ı kullanıyorum
# .. encode_plus : Bert modeline vereceğim textleri tokenleştirirken encode_plus ı kullanacağım
# .. encode      : Sadece her bir vektörümü(textimi) modele verirken sabit uzunluğu ne olsun? sorusuna cevap için kullanıyoruz

"""
class chat: max_token listesi inceleme
import numpy as np
from scipy import stats
arr = np.array(max_token)
 
print("Descriptive analysis")
print("Document Size \t=", arr.shape[0])
print("Doc Token Count\t=", arr)

# measures of central tendency
print("Measures of Central Tendency")
print("Mean \t\t=", arr.mean())
print("Median \t\t=", np.median(arr))
print("Mode \t\t=", stats.mode(arr)[0][0])

# measures of dispersion
print("Measures of Dispersion")
print("Minimum \t=", arr.min())
print("Maximum \t=", arr.max())
print("Range \t\t=", arr.ptp())
print("Variance \t=", arr.var())
print("Standard Deviation =", arr.std())
"""

# Her birini 160 tokene sabitlemeliyiz
tokens = tokenizer.encode_plus(sentence, max_length=160, truncation=True, padding='max_length', add_special_tokens=True)
tokens
# max_length=160 : Her bir yorumum hangi uzunlukta olacak bunu atadıktan sonra bunu padding in karşısına yazacağız "max_length" i. Yoksa hata alırsınız
# input_ids,token_type_ids, attention_mask i 160 boyuta sabitledi
# truncation=True : Kesme işlemi gerekirse yap
# padding='max_length' : max_length ne ise bunun kendisini yazıyoruz. Yani önce max_length e 160 atıyoruz sonra burada padding de yazıyoruz
# .. padding = True ya da padding = 160 vs derseniz hata alırsınız
# add_special_tokens=True : Özel tokenlerimizi başa ve sonra ilave et
# !!! Burada input_ids, token_type_ids ve attention_mask in her birini 160 boyutlu olacak şekilde 0 larla doldurdu
# !!!Tokenleştirme işlemini bu kod ile yapmış olduk
# class chat soru: Hocam burada 0 ları sona ekledi?
# !!!Johnson H: DL de başa eklerken burada sona ekliyor

# Her bir yorumu vektörlere dönüştürdük. Ama benim 24300 yorumum var. Şimdi Bu elde ettiğim vektörleri matrise dönüştüreceğim
# Her bir yorumu alıp bir matris içerisinde birleştireceğiz
seq_len = 160        # Hepsi 160 boyutlu olsun
num_samples = len(X) # gözlem sayım, satır sayım, yorum sayım # 243000

Xids = np.zeros((num_samples, seq_len))
Xmask = np.zeros((num_samples, seq_len))
# 0 lardan oluşan 2 ayrı matris oluşturuyoruz burada. 160 feature, 243000 satırlık matris
# Xids = input_ids  ve Xmask= attention_mask
# class chat soru: token_type ids i kullandığımız için mi eklemedik. Kullansak yapacak mıyız?
# Johnson H: Kullanmadığımız için almadık ama translation yapacaksanız onun içinde bir matris oluşturmanız gerekirdi
# .. token_type ids zaten "0" lardan oluşacağı için hep modele bir faydası yok

# Xids[:2]  # Şimdi input_ids leri yukarıdan çekeceğiz ve bu matrisin ilk sırasına 
# .. yerleştireceğiz( 2. köşeli parantez( "[" ) ve 3. ters köşeli parantez( "]" ) arasına) 
# Xmask[:2] # Aynı şekilde bunun içinde attention_mask leri alıp bunun ilk sırasına yerleştireceğiz
# Yani datamda kaçıncı yoruma denk geliyorsa, o yoruma karşılık o input_ids ve attention_mask leri alacağım ve
# .. bu matrislerin içerisine teker teker yerleştireceğiz. Yani vektörlerden matrise dönüşümü bu şekilde yapacağız
# Her bir yorumu bir for dönügüsü oluşturacağız önce for döngüsü ile teker teker her bir cümleyi
# .. input_ids ve attention_mask vektörlerine dönüştüreceğiz daha sonra da oluşturduğumuz 0 lardam oluşan matrislerin
# .. datamda ilgili satırlara(kaçıncı yoruma denk geliyorsa) sırayla içine vereceğiz. Bunu nasıl yapacağız bakalım altta

print(tokens['input_ids'])      # "tokens" e ait 160 boyutlu input_ids vektörümüzü aldık
print(tokens['attention_mask']) # "tokens" e ait 160 boyutlu attention_mask vektörümüzü aldık

"""
# Bu çektiklerimizi ilgili matrisimizdeki ilgili(kaçıncı yoruma denk geliyorsa) satıra ilave ediyorum. Xids[0]: Xid lerin ilk satırları
Xids[0] = tokens['input_ids']         # Xids[0]: Xids nin 0. satırı . ilk satıra tokens['input_ids']  bilgilerini ilave etti burada
Xmask[0] = tokens['attention_mask']
Xids[0]
Xmask[0]
"""

# Üsttekileri bütün corpus a uygulayalım
def transformation(X):
  # set array dimensions
  seq_len = 160        # matrisin sütun sayısı
  num_samples = len(X) # matrisin satır sayısı
  # initialize empty zero arrays
  Xids = np.zeros((num_samples, seq_len))
  Xmask = np.zeros((num_samples, seq_len))   
  for i, phrase in enumerate(X):
      tokens = tokenizer.encode_plus(phrase, max_length=seq_len, truncation=True, padding='max_length', add_special_tokens=True)  
      # assign tokenized outputs to respective rows in numpy arrays
      Xids[i] = tokens['input_ids']
      Xmask[i] = tokens['attention_mask']
  return Xids, Xmask
# Xids[i] ye karşılık gelen tokens['input_ids'] mizi 
# Xmask[i] ye karşılık gelen tokens['attention_mask'] mizi birleştirdik
# enumerate: o yorumun, kaçıncı yoruma denk geldiğini tespit edeceğiz enumerate yardımıyla
# Bu fonksiyonu kullanarak bütün textimizi vektörlerden matrislere dönüştürebiliriz

# train ve test setlerimizi Xids ve Xmask setlerine dönüştürelim
Xids_train, Xmask_train = transformation(X_train) 
Xids_test, Xmask_test = transformation(X_test)
# Burada hem train setinin hem de test setinin input_ids ve attention_mask leri ilgili sayısal verilerle doldurduk

print("Xids_train.shape  :", Xids_train.shape)  # (219147, 160)
print("Xmask_train.shape :", Xmask_train.shape) # (219147, 160)
print("Xids_test.shape   :", Xids_test.shape)   # (24350, 160)
print("Xmask_test.shape  :", Xmask_test.shape)  # (24350, 160)

y_train # Labelımızı da matrise dönüştürmemiz lazım. vektör olarak bırakamayız

labels_train = y_train.reshape(-1,1)  # -1: satır sayısı ne ise kendin ayarla, 1: feature/sütun sayısı
labels_train
# labelımızı tek boyutlu bir matrise dönüştürmüş olduk

y_test

labels_test = y_test.reshape(-1,1)
labels_test

"""
# Burayı not olarak yazdık 
from tensorflow.keras.utils import to_categorical
l = np.array([0, 1, 3, 5, 4, 2])
to_categorical(l, 6) 
# Label ınız 2 den fazla class a aitse burada olduğu gibi(burada 6 tane var([0, 1, 3, 5, 4, 2]))
# "from tensorflow.keras.utils import to_categorical" u kullanarak l = y_test veya l=y_train olduğunu varsayalım
# .. bunu veriyorsunuz to_categorical içerisine, o size istediğiniz matris boyutuna getiriyor
# .. Eğer bunu üstte benim modelim için kullansaydık matrisim 2 boyutlu olurdu ve sigmoid yerine softmax kullanmak zorunda kalırdık
# .. softmax maliyet olarak geri dönerdi bize ve bir de "argmax" vs yaparak en yüksek olasılığı hangisi verdi falan onu çekmeniz lazım
# .. binary de bunu kullanmanıza gerek yok. binary de reshape(-1,1) kullanın
"""

##### Transformation Matrix to Tensor
# Buraya kadar her bir yoruma ait 3 tane vektörümüzü elde ettik. Burada classification yapacağımız için
# .. input_ids ve attention_mask a ihtiyacımız vardı. Sonra matrislerimizi elde ettik
# .. son aşamada bu matrisleri bir araya getirip tek bir tensorda birleştireceğiz
# Tensor: Birden fazla matrisin bir araya gelerek oluşturduğu yapı
# !!! Xids ve Xmask matrisleri nenim iputlarım, labels_train ve labels_test de benim outputlarım. Bunların hepsini birleştirip tensora dönüştürmem lazım
# Bert modelleri bizden bunları tensor olarak istiyor
import tensorflow as tf
dataset_train = tf.data.Dataset.from_tensor_slices((Xids_train, Xmask_train, labels_train))
dataset_train
# Xids_train, Xmask_trainn, labels_train i(Bu üçünü) tensor içerisinde birleştirecek
# Yani her birini tensor yapacak(3 tane tensor döndürecek).
# Output:<TensorSliceDataset element_spec=(TensorSpec(shape=(160,), dtype=tf.float64, name=None), TensorSpec(shape=(160,), dtype=tf.float64, name=None), TensorSpec(shape=(1,), dtype=tf.int64, name=None))>
    
# Aynı şeyi test seti için de yapalım
dataset_test = tf.data.Dataset.from_tensor_slices((Xids_test, Xmask_test, labels_test))
dataset_test
# Output: <TensorSliceDataset element_spec=(TensorSpec(shape=(160,), dtype=tf.float64, name=None), TensorSpec(shape=(160,), dtype=tf.float64, name=None), TensorSpec(shape=(1,), dtype=tf.int64, name=None))>

# Bunları tensorlere dönüştürdük ama benim modelim bir modele text e verdiğim zaman bert şunu der,
# .. bana aynı boyutta(160) 2 tane tensor verdin. Bunlar benim için aynı şey.
# .. Bunların hangisi input id, hangisi attention mask diye sorar. Bunu söylemezsen işlem yapamam der
def map_func(Xids, Xmask, labels):
    # we convert our three-item tuple into a two-item tuple where the input item is a dictionary
    return {'input_ids': Xids, 'attention_mask': Xmask}, labels
# input_ids, attention_mask: Bu yazdığımız isimler yukarı tokenizer(tokenleri ayırma) işlemi yaptığımız isimler ile aynı olmalı
# input: {'input_ids': Xids, 'attention_mask': Xmask}
# !!! NOT: Bunu mutlaka return de süslü parantez içerisinde belirliyoruz. Dışarda kalan benim output um oluyor(Bunu model algılıyor)
# Burada input ve output ayrımını yapıyoruz yani

# then we use the dataset map method to apply this transformation
dataset_train = dataset_train.map(map_func)
dataset_test = dataset_test.map(map_func)

dataset_train
# Output: <MapDataset element_spec=({'input_ids': TensorSpec(shape=(160,), dtype=tf.float64, name=None), 'attention_mask': TensorSpec(shape=(160,), dtype=tf.float64, name=None)}, TensorSpec(shape=(1,), dtype=tf.int64, name=None))>
# .. dışarda kalan zaten output umuz bir daha output yazmamıza gerek yok
dataset_test
# Output: <MapDataset element_spec=({'input_ids': TensorSpec(shape=(160,), dtype=tf.float64, name=None), 'attention_mask': TensorSpec(shape=(160,), dtype=tf.float64, name=None)}, TensorSpec(shape=(1,), dtype=tf.int64, name=None))>

# Modelin istediği forma getirdik. Şimdi modelimizi tensorlere ayırdık fakat bu tensorları biz modele verirken
# .. batch_size lara bölerek vermemiz lazım

##### Batch Size and Shuffle Train Dataset
# !!! Johnson H: Bu kısım önemli
# Bert modelini geliştiren developerların kendi makalesindeki bilgi : batch_size: 16, 32 deneyin(Overfitting durumuna göre)
# Johnson h: Overfitting varsa ancak overfit varsa 64 vs de deneyebilirsiniz ama gerek kalmıyor.
# .. Ancak batch_size küçük oldukça model daha iyi öğrenmeye meyillidir. Batch size büyüdükçe öğrenme sınırlıdır

batch_size = 32 # 16, 32
train_ds = dataset_train.batch(batch_size) # Gözlemlerin yeri sabit olduğundan bunu metricler için kullanacağız. 
val_ds = dataset_test.batch(batch_size)
# Hem train seti ve validation setini 32 li paketlere böldük ama altta extra bir şey yaptık(NOT: dataseti kendi içinde karılmaz üsttekilerde. Alttaki açıklamaya bkz.)
length = len(X_train)
train_ds2 = dataset_train.shuffle(buffer_size = length, reshuffle_each_iteration=True).batch(batch_size) 
# shuffle: Tensorlar arasında her epoch da karılma işlemi olmasını istiyorsam(batch size lar kendi içinde karılıyor)
# .. bunu kullanıyorum(normalde batch size da karılma oluyordu ama burada
# .. tensorlerde karılma işlemi olmasını istiyorsak bu shuffle fonksiyonunu kullanıyor)
# buffer_size = length : karıştırılma işlemi nasıl yapılsın. Kaçlı paketlere bölüp karıştırayım. 
# .. Dökümanda datanın kendi boyutuna eşit sayı verirseniz en mükemmel karışımı alırsınız diyor. O yüzden buffer_size a datanın kendi boyutunu verdik
# reshuffle_each_iteration=True: İç iterasyon. Her iterasyonda dataseti kendi içinde karılsın mı 
# Buffer size boyutu train datasının boyutuyla aynı olduğunda eğitim aşamasında train datası en iyi şekilde karıştırılır.
# Eğitim aşamasında train_ds2(Bunu fit içine vereceğiz), predict aşamasında train_ds kullanacağım(Çünkü train_ds2 yi(yani karılmış olanı
# .. predict te kullansaydık predict te kötü sonuçlar saçmasapan gelir.(Aşağıda göreceğiz))

##### Creating Model
def create_model():
    from transformers import TFAutoModel  # NOT:AutoModel başına TF yazabiliyoruz. Notebook u dışardan göre biri Tensorflow un Tensorlarını mı kullanmış bilsin diye başına TF yazdık
    model = TFAutoModel.from_pretrained("dbmdz/bert-base-turkish-uncased")
    input_ids = tf.keras.layers.Input(shape=(160,), name='input_ids', dtype='int32')
    attention_mask = tf.keras.layers.Input(shape=(160,), name='attention_mask', dtype='int32')
    embeddings = model.bert(input_ids=input_ids, attention_mask=attention_mask)["pooler_output"] #[1]
    x = tf.keras.layers.Dense(160, activation='relu')(embeddings)  # dense layer a word_embedding bilgilerini aktar. Aktaracağım layer ı da tuple içinde ("embeddings") yazıyorum.(functional API tarzı(Önceden sequential tarzı ile yazıyorduk))
    x = tf.keras.layers.Dropout(0.1, name="dropout")(x) # 0.1 # x deki bilgileri dropout a aktardık
    y = tf.keras.layers.Dense(1, activation='sigmoid', name='outputs')(x) # Üstteki x bilgileri de buraya aktardık. Her bir token hangi tokenlerle kullanılmış bu kısımda görüyor. Eğer son bir revize yapılacaksa bu kısımda yapılıyor
    return tf.keras.Model(inputs=[input_ids, attention_mask], outputs=y) # inputlarımı ve outputumu veriyoruz

# Modeli tanımlıyoruz burada. Tensorflow un kendi sayfasında kullandığı modeli aldık yapıştırdık buraya direk
# tokenizer ı "uncased" üzerinden yapmıştık o yüzden modeli de "uncased" üzerinden kuruyoruz burada. Yoksa skorlar çok farklı olur
# NOT: import ve model tanımlamayı her zaman fonksiyon içinde yapın. Eğer dışarda yaparsanız TPU çekirdeklerinden faydalanma aşamasında hata alırsınız
# Burada tensorflow un "input" layerlarını kullanacağız. Hatırlarsanız DL derslerinde de input layerlarımız vardı ama onu Dense layerlar içerisinde input
# .. boyutlarını belirtiyorduk o yüzden bizim bir daha input layerlarını kullanmamıza gerek kalmıyordu. Fakat burada;
# .. tensorflow developerları input layerlarımızı kullanmamızı istiyor. Modele verilecek inputların boyutlarını istiyor
# .. Birisi input_ids, diğeri attention_mask ..Her biri(tensorler) 160 boyutunda olacak.
# .. Ayrıca isimleri hangi tensor a aitse bunu belirtmemiz lazım "name" de. "map_func" daki isimler ile aynı isimler olmalı buradakiler
# .. yoksa hata alırsınız
# .. dtype='int32': Bert modelleri "int32" de çalışır(Tensorlerim normalde Float tı(Üstte dataset_train ve dataset_test çıktısına bakabiliriz teyit etmek için(satır 3123 ve 3126)))
# .. bu kısımda artık modeldataya gelen tensorler üzerinden hangisine input_ids, hangisine attention_mask muamelesi yapacağını biliyor
# embeddings: Model üzerinde eğitilmiş hazır olan tokenleri çekme kısmı.
# .. input_ids = input_ids, attention_mask=attention_mask: dataya input u verdik ve her bir tokene karşılık gelen word_embeddingleri model üzerinden çekiyoruz burada
# .. ["pooler_output"]: model3.summary() kısmında "pooler_output" denen bir katman var. Bu katmanı biz
# .. bert modellerin üzerinde eğitildiği word embedding bilgilerini tuttuğu katmandır. Yani ["pooler_output"] yaptığımda diyorum ki
# .. ben bert modelin ilgili layerına(["pooler_output"] a) geldim. Şimdi bu layerda input_ids lere karşılık gelen word embeddingleri çekeceğiz
# .. [1]: Bu bert modelimizin 1. katmanı olduğu için "pooler_output" yerine "1" de yazabilirsiniz
# x = tf.keras.layers.Dense(160, activation='relu')(embeddings) : "embeddings" kısmında(embeddings = model.bert(inp...)) elde ettiğim word embeddingleri
# .. dense layer a aktar diyoruz burada. Normalde bu sequential yapısından farklı bir yapı.
# .. DL de 2 tane yapı vardı biri "sequential", diğeri "functional API". Arasındaki fark şudur;
# .. "sequential" da bir layerdan bir sonraki layer a bilgiler aktarılırken "add" kullanılır
# .. "functional API" de aktaracağım bilgileri hangi katmana aktaracaksam o katmana bir tuple açıp tuple içerisinde bunu(embeddings) yazıyoruz.
# .. Yani add kullanmak yerine bu şekilde aktarım yaptık burada
# .. Peki burada neden dense layer a aktarıyoruz. Modelim zaten bütün tokenler arasındaki anlamsal ilişkileri çok iyi biliyor
# .. tekrardan dense layer a neden aktarıyorum ki ? en sondaki dense layer a aktarayım activation='sigmoid' olsun ve tahminlerimi alayım gibi düşünebilirsiniz
# .. ancak burada developerların düşüncesi şu: Benim tokenlerim en iyi şekilde anlamsal ilişkileri kurmuşlar ana sonra aşamada dense layer içerisinde
# .. bütün tokenlerin birbirleriyle kullanımlarını son bir kez görmek istiyor. Acaba benim extra öğrenmem gereken bir yapıda olabilir onu da göreyim diyor
# .. Yani son cilayı çekme kısmı burası. Bu dense layerda hangi token hangi tokenle beraber kullanılmış burada görüyor ve sonuç olarak revize yapacaksak 
# .. bu(x = tf.keras.layers.Dense(160, activation='relu')(embeddings)) kısımda yapıyoruz ve bir dropout a daha tabi tutup sonra tahminlerimi alıyorum
# x = tf.keras.layers.Dropout(0.1, name="dropout")(x) # 0.1(Dense layerdaki bilgileri (maximum) %10 oranında sönümlendir) # x deki bilgileri dropout a aktardık
# .. Developerlar tarafından tavsiye edilen yöntem, dropout a 0.1 yaparak overfit i engelleyin diyorlar. Johnson H: ben bazen 0.2 yi deneyip engelledim 
# y = tf.keras.layers.Dense(1, activation='sigmoid', name='outputs')(x) # x = tf.keras.layers.Dropout(0.1, name="dropout")(x) layerındaki bilgileri 
# .. tahmin alacağım(son dense) layera aktar. Yukarda y için tek boyutlu matrisler oluşturmuştuk, eğer
# .. Üstte "to_categorical" kullansaydık son kısımda "1, activation='sigmoid' " yerine  "2, activation='softmax' " yazmalıydık
# return tf.keras.Model(inputs=[input_ids, attention_mask], outputs=y) : Son aşamada modelimizi kuruyoruz input ve outputlarımızı vererek
# outputs=y : buradaki y, bir nevi y_pred ler yani bunu eğitim aşama tahmini y ler alacak. Bu tahmini y leri gerçek  y değerleriyle fit aşamasında
# .. karşılaştırarak ve loss u hesaplayacak. Yani buradaki(outputs=y) y ler gerçek "y" lerim değil

# Burası TPU çekirdeklerinin hızından faydalanma kısmı
with strategy.scope():                              # TPU çekirdekleri kapsamında(scope) hangi işlemleri yapayım
  optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5) #3e-5, 5e-5 # NOT: learning_rate=2e-5 3e-5, 5e-5 makalede geçen öneriler
  loss = tf.keras.losses.BinaryCrossentropy()
  recall = tf.keras.metrics.Recall()
  model3 = create_model()
  model3.compile(optimizer=optimizer, loss=loss, metrics=[recall])
  # NOT: initializing TFBertModel: ['mlm___cls', 'nsp___cls'] uyarısını dikkate almayın


model3.summary()
# 110,740,545 parametre ile işlem görmüş data. 

history = model3.fit(train_ds2, validation_data= val_ds, class_weight= {0:1, 1:4}, epochs=1) 
# class_weight= {0:1, 1:4} : ağırlıklandırmayı düşük yaparsanız güzel sonuçlar elde edebiliyorsunuz BERT de normalde sınıflar arasında 19 kat fark var
# .. olumlu yorumlara 1 kat, olumsuz yorumları 4 kat ağırlıklandırma yaptık
# epoch= 2 or 3 # Dökümanda "2 ve ya 3 den daha fazla epoch yaptırmayın overfit e gider" der. Overfit i epochs=1 ile engellediğimiz için 1 epoch yaptık
# epoch=1 de karışma işlemi olmayacak/gerek kalmayacak??? Evet ama her zaman böyle olmayacak üstteki shuffle ı o yüzden gösterdik
# .. milley epoch = 10 falan yapıyor. Yanlış
# Bu işlem, CPU : 180 saat, GPU: 2-3 saat, TPU: 10 dk sürüyor

##### Model evaluation
from sklearn.metrics import classification_report, confusion_matrix
y_pred = model3.predict(val_ds) >= 0.5
print(classification_report(y_test, y_pred))
# Bu kadar dengesiz bir datada 0.71 precision ı hiç bir metod da yakalayamazsınız BERT haricinde.
# .. Önceki skor un 2 katı neredeyse
# recall 0.86 gelmiş(0.91 e kadar çıkıyor skor. Bir kaç defa daha çalıştırırsanız)
# Modeli her çalıştırdığınızda bir değişkene kaydedelim ki o skorları kaybetmeyelim
# train test arasında farklarda yok çok yani overfitting yok.
# Johnson H: 0.74 e 0.91 skorlarını yakalıyor(0.71      0.86 ) ama ben modeli 10 defa çalıştırdım kaydetmeyi unuttuğum için bu skorları görüyoruz
# .. modeli her çalıştırdığınızda değişkeni kaydedin mutlaka
"""
761/761 [==============================] - 35s 38ms/step
              precision    recall  f1-score   support

           0       0.99      0.98      0.99     22982
           1       0.71      0.86      0.78      1368

    accuracy                           0.97     24350
   macro avg       0.85      0.92      0.88     24350
weighted avg       0.98      0.97      0.97     24350
"""

y_train_pred = model3.predict(train_ds) >= 0.5
print(classification_report(y_train, y_train_pred)) 
"""
6849/6849 [==============================] - 205s 30ms/step
              precision    recall  f1-score   support

           0       0.99      0.98      0.99    206839
           1       0.76      0.91      0.83     12308

    accuracy                           0.98    219147
   macro avg       0.88      0.95      0.91    219147
weighted avg       0.98      0.98      0.98    219147
"""
######
"""
# y_train_pred2 = model3.predict(train_ds2) >= 0.5
# print(classification_report(y_train, y_train_pred2)) 
# prediction da train_ds2 kullansaydık, sonuçlarımız böyle saçmasapan gelirdi

6849/6849 [==============================] - 195s 28ms/step
              precision    recall  f1-score   support

           0       0.94      0.93      0.94    206839
           1       0.06      0.07      0.06     12308

    accuracy                           0.88    219147
   macro avg       0.50      0.50      0.50    219147
weighted avg       0.89      0.88      0.89    219147
"""

"""
Unweighted model
# Ağırlıklandırma yapmadan bile skorlarımız çok güzel
TEST SET

            precision    recall  f1-score   support

       0       0.99      0.97      0.98     22987
       1       0.79      0.80      0.80      1365
TRAIN SET

            precision    recall  f1-score   support

       0       1.00      0.97      0.99    206825
       1       0.85      0.85      0.85     12311

# Ağırlıklandırma yapmadan DL de skorlarımız alttaki gibiydi
# Test: 0,61 0.75
# Train: 0.85, 0.90 idi ... Burada;
# Test : 0.79      0.80
# Train: 0.85      0.85  oldu
"""
# Bu kadar dengesiz bir datada(%96 ya, %4) skorları ML, DL de elde edemezsiniz. O yüzden BERT i tercih edin

from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt
y_pred_proba = model3.predict(val_ds)
PrecisionRecallDisplay.from_predictions(y_test, y_pred_proba)
plt.show();
# Johnson H: Bu kodu Allen hocadan bulduk. Hem model performansını görüyoruz hem de görselleştiriyoruz
# .. Model performansı %88, DL de %72 civarındaydı

model3.save("/content/drive/MyDrive/sentiment_model.h5") # Kendi drive ınızın adresini yazabilirsiniz

from tensorflow.keras.models import load_model
model4 = load_model('/content/drive/MyDrive/sentiment_model.h5')

##### PREDICTION
# initialize tokenizer from transformers
# Model eğitilirken tensorlerle eğitildi. Yani input_ids nin ve attention_mask in tensorlara dönüştürülmüş versiyonları vardı
# .. o yüzden burada output a ihtiyaç yok sadece tahmin alacağız. Burada ihtiyacımız olan input_ids ve attention_mask i tensor olarak çekmek
# NOT: Eğitim aşamasında kullandığınız tokenizer ne ise bunu prediction aşamasında da kullanıyoruz. Yukarda yazdık diye burada yazmamazlık yapmıyoruz
# .. Burada neden bahsediyoruz? -->  tokenizers = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")
from transformers import AutoTokenizer, TFAutoModel
tokenizers = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")
def prep_data(text):
    # tokenize to get input IDs and attention mask tensors
    tokens = tokenizers.encode_plus(text, max_length=160, truncation=True, padding='max_length', add_special_tokens=True, return_tensors='tf')  
    return {'input_ids': tokens['input_ids'], 'attention_mask': tokens['attention_mask']} 
# return_tensors='tf': Her bir yorumu otomatik olarak tensor a dönüştür
# .. Tensor olarak dönüştürdüğüm için return dekiler vektör olarak değilde tensor olarak gelecek ve prediction için hazır olmuş olacak

review1 = "çok beğendim herkese tavsiye ederim"
review2 = "süper ürün aynı gün elime geçti"
review3 = "büyük bir hayal kırıklığı yaşadım bu ürünü bu markaya yakıştıramadım"
review4 = "kelimelerle tarif edilemez"
review5 = "tasarımı harika ancak kargo çok geç geldi ve ürün açılmıştı hayal kırıklığı gerçekten"
review6 = "hiç resimde gösterildiği gibi değil"
review7 = "kötü yorumlar gözümü korkutmuştu ancak hiçbir sorun yaşamadım teşekkürler"
review8 = "hiç bu kadar kötü bir satıcıya denk gelmemiştim ürün siparişimi iptal ediyorum"
review9 = "tam bir fiyat performans ürünü"
review10 = "beklediğim gibi çıkmadı"
review11 = "on numara beş yıldız"
review12 = "bu kargoyu biran önce bırakın. yoksa üründe sıkıntı yok"
review13 = "televizyonun görüntü kalitesi çok kötü, dün akşam evde arkadaşlarla toplandık. yedik, içtik, sohbet ettik. Sonra televizyonda Türkiye - İngiltere maçını seyrettik. "
review14 = '1 gün gibi kısa bir sürede elime geçti. Ve bu fıyata süper bir ürün tavsiye ederim. Lakin eli büyük olan kişiler daha büyük modelini tercih edebilirler ortaboy ürün. Teşekkürler '
review15 = "alınca anlarsın anyayı konyayı"
review16 = "çöpe atacak paran varsa al"
review17= "Telefon çok da kötü değil"
review18 = "al da gününü gör"
review19 = "Ürün harika ama satıcı ve kargo berbat"
review20 = "kargo süper ama ürün berbat"
review21 = "Aldigim TV cok kaliteli diye dusunmustum, sonradan cok da iyi bir TV olmadigini dusundum, ama neyse yine de memnunum."
review22 = "😊"
review23 = ":)"
review24 = "I ❤️ you"
review25 = 'telefon kesinlikle bu parayı hak etmiyor. yeni hiç bir özelliği yok. ancak küçük ekran sevenler için iyi bir telefon'
review26 = 'müthiş bir telefon ama bu parayı hak etmiyor'
reviews = [review1, review2, review3, review4, review5, review6, review7, review8, review9, review10, review11, review12, review13, review14, review15, review16, review17, 
           review18, review19, review20, review21, review22, review23, review24, review25, review26] 

probs = []
for i in reviews:
  in_tensor = prep_data(i)  # prep_data(i): teker teker fonksiyona yorumlarımı veriyorum, in_tensor: tensor a dönüşmüş hali gelecek
  prob = model4.predict(in_tensor)[0][0]  # [0][0]: "model4.predict(in_tensor)" 2 boyutlu döndüğü için, bunun içindeki değeri almak için "[0][0]" yaptık
  probs.append(prob)
  # her aşamada aldığım olasılığı probs içine atadık

in_tensor
# in_tensor istediğim formata gelmiş: 1.tensora dönüşmüş. 2.boyut 1 e 160, 3.dtype='int32'(<tf.Tensor: shape=(1, 160), dtype=int32, nu...devam ediyor)

# model4.predict(in_tensor)       # Sonucu 2 boyutlu döndürüyor
# model4.predict(in_tensor)[0][0]  # 2 boyutlu array in içerisinden bu değri çekmek için [0][0] yaptık sona

probs # Her bir yoruma ait olasılıklar

# Olasılıkları sınıflara dönüştürelim
classes  = (np.array(probs) >= 0.5).astype("int")
classes

my_dict = {"Review":reviews, "prob":probs, "classes":classes}

pd.set_option('display.max_colwidth', None)
pd.DataFrame(my_dict)
# Yorumlarımızın, olasılıkları ile birlikte hangi sınıfa ait olduğunu görüyoruz
# 4. indexte hayal kırıklığı nın olumsuzda ağırlıklı bir keyword olduğunu öğrenmiş model "harika" geçmesine rağmen olumsuz olarak tahmin etmiş
# 16. index çöğe atacak paran varsa al: güzel yakalamış
# 17. index telefon çok da kötü değil: Bu ortada bir yorum biraz. Buna olumlu demiş neden? çünkü yapılan yorumlarda "çok da kötü değil" i olumlu yorum
# .. olarak görünce model olumlu döndürmüş
# Ürün harika ama satıcı ve kargo berbat : ürüne yapılan yorumu daha çok dikkate almış, satıcı ve kargoya yapılan yorumdan ziyade 
# kargo süper ama ürün berbat: ürüne yapılan yorumu dikkate almış
# emoji ye yani  ":)" e olumlu gelmiş 
# class chat soru: ben de bazı sonuçlar değişik hocam
# Johnson H: Katsayılar farklı yerlerde başladığı için ve karma yaptığı için tokenlere atanan 
# .. katsayılarda farklılıklar olabilir. Ayrıca modelimizde ağırlıklandırma yaptığımız için
# .. bazı olumlu yorumları da olumsuz görebiliriz çünkü olumsuzları tahmin etmeye ağırlık ver demiştik
# NOT: Karıştırma işlemi corpus içerisindeki tokenlere göre yapılmıyor, labellara göre yapılıyor. O yüzden
# .. 2 tane olumsuz(train de olmayan) yorum test datasında kalırsa bunu model öğrenemez

# al da gününü gör, "anyayı konyayı"? neden bilemedi? Altta bakacağız
count=0
for i in X_train:
  if "gününü gör" in i.lower():
    count+=1
print(count)
# gününü gör: Data bunu train de görmemiş hiç. Model bunun olumsuz bir yorum içinde kullanıldığını öğrenememiş

count=0
for i in X_train:
  if "anyayı konyayı" in i.lower():
    count+=1
print(count)
# anyayı konyayı: Data bunu train de görmemiş hiç . Model bunun olumsuz bir yorum içinde kullanıldığını öğrenememiş

count=0
for i, j in enumerate(X):
  if "çöpe atacak paran" in j.lower() and y[i]==1:
    count+=1
    print(i)
print(count)

X[158068]

# class chat: tokenizationda türkceyi yükledik ama ingilizce yorumu da bildi, ingilizce default mu acaba,baska dilde yorum olsaydi onda da tahmin yapabilir miydi?
# Johnson H: Bert Turk, google ın sağladığı datalar üzerinde eğitmiş ama ingilizce yorumu yakalamış olması ile ilgili net bir şey söyleyemem
# .. ama bert-turk ün eğitiminin tamamen türkçe üzerinde olduğunu biliyorum. 
# .. class chat cevap : İngilizce olduğundan değil de data içerisinde var olduğundan olabilir(johnson H: bu da olabilir evet)
# Johnson H: Bundan sonra hep BERT modellerini ya da hazır modelleri tercih etmemiz daha iyi olur.

#%% NLP-8 PROJECT SOLUTION(LAST SESSION)























