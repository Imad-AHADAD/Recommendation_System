from sklearn.preprocessing import LabelEncoder
import nltk
import re
from nltk.corpus import stopwords
# nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd



def preprocess_data(ds):

  # Encoding the Features that have ordinal values (LabelEncoder)
  encoder = LabelEncoder()
  ds['Magnitude_Risk'] = encoder.fit_transform(ds['Magnitude_Risk'])
  ds['Impact'] = encoder.fit_transform(ds['Impact'])

  # Data cleaning for encoding (stopwords.words)
  STOP_WORDS = nltk.corpus.stopwords.words()

  def clean_sentence(val):
    "remove chars that are not letters or numbers, downcase, then remove stop words"
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', val).lower()
    sentence = sentence.split(" ")
    for word in list(sentence):
        if word in STOP_WORDS:
            sentence.remove(word)
    sentence = " ".join(sentence)
    return sentence

  def clean_dataframe(data):
    cols = ['Requirements', 'project_Category', 'Requirement_Category', 'Risk_Target_Category', 'Dimension_Risk', ]
    for col in cols:
        data[col] = data[col].apply(clean_sentence)
    return data
  
  STOP_WORDS = nltk.corpus.stopwords.words()
  ds = clean_dataframe(ds)
  # Encoding the Features with one word (OneHotEncoder)
  columns_one_hot = ['project_Category', 'Requirement_Category','Risk_Target_Category', 'Dimension_Risk']
  # Use get_dummies to convert categorical variable into dummy/indicator variables
  dummies = pd.get_dummies(ds[columns_one_hot], drop_first=True)

  # Concatenate the dummy variables with the original DataFrame
  df_with_dummies = pd.concat([ds, dummies], axis=1)

  df_with_dummies.drop(columns_one_hot, axis=1, inplace=True)

  
  # Droping the row with fix_cost=="?"
  index_to_drop = df_with_dummies.loc[df_with_dummies["Fix_Cost"]=="?",].index
  df_with_dummies.drop(index_to_drop,axis=0,inplace=True)
  # casting the fic_cost to int 
  df_with_dummies["Fix_Cost"] = df_with_dummies["Fix_Cost"].astype(int)

  # Encoding the Requirements column with CountVectorizer

   # Multiple documents
  text = df_with_dummies["Requirements"] 
  # create the transform
  vectorizer = CountVectorizer()
  # tokenize and build vocab
  vectorizer.fit(text)
  # encode document
  vector = vectorizer.transform(text)

  # This creates a new DataFrame object with columns given and the values id vactor values
  df = pd.DataFrame(columns=sorted(vectorizer.vocabulary_), data=vector.toarray())
  # cancate the actual data set (df_with_dummies) with df the replace theRequirements column
  df_encoded = pd.concat([df_with_dummies, df], axis=1)
  # Droping the requirement column because it's beenn replaced by df
  df_encoded.drop("Requirements",axis=1,inplace=True)

  # Droping rows with Nan values after data changing
  df_encoded.dropna(inplace=True) 
  
  # Our last comumn will be our target
  columns = list(df_encoded.columns)
  columns.remove('Risk_Level')
  columns.append('Risk_Level')
  df_encoded = df_encoded[columns]
    
  # returning the final data after data preprocessing
  return df_encoded