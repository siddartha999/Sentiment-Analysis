import nltk
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
nlp = spacy.load('en_core_web_sm')
from textblob import TextBlob
from pattern.en import sentiment

nltk.download('stopwords')

df = pd.read_table('data.txt')
df.head()

f = open('data.txt', 'r')
content = f.read()
f.close()
print(content)


sentence=[]
tokens = nlp(content)
for sent in tokens.sents:
    sentence.append((sent.text.strip()))

print(len(sentence))

textblob_sentiment=[]
for s in sentence:
    txt= TextBlob(s)
    a= txt.sentiment.polarity
    b= txt.sentiment.subjectivity
    textblob_sentiment.append([s,a,b])
    
df_textblob = pd.DataFrame(textblob_sentiment, columns =['Sentence', 'Polarity', 'Subjectivity'])
df_textblob.head()

sns.displot(df_textblob["Polarity"], height= 5, aspect=1.8)
plt.xlabel("Sentence Polarity")


sns.displot(df_textblob["Subjectivity"], height= 5, aspect=1.8)
plt.xlabel("Sentence Subjectivity")

tokenizer = nltk.tokenize.RegexpTokenizer('w+')
tokens = tokenizer.tokenize(content)
len(tokens)




plt.subplots(figsize=(16,10))
stopwords = set(STOPWORDS)
stopwords.update(["hello", "thanks", "happening", "see", "now", "going", "want", "really", "well", "around", "able", "year", "continue"
  "event", "make", "topics", "lot", "easier", "platform", "show", "see", "continue", "easier", "around", "service", "looking", "tab", "sure",
  "start", "use", "little", "said", "will", "something", "debate", "reset", "apply", "look", "within", "much", "right"])
wordcloud = WordCloud(
                          background_color='black',
                          max_words=100,
                          width=1400,
                          height=1200,
                      stopwords=stopwords
                         ).generate(content)
plt.imshow(wordcloud)
plt.title('Word Cloud')
plt.axis('off')
plt.show()