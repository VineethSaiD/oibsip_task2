{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "a83bee0c-a7d3-427d-8a7c-e0ca774d84c0",
   "metadata": {},
   "outputs": [],
   "source": [
    "#import libraries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import re"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "06cc44eb-4c60-4e7f-87dc-17948e62c426",
   "metadata": {},
   "outputs": [],
   "source": [
    "# sms spam data loading and encoding dataset\n",
    "df= pd.read_csv(\"C:\\\\Users\\\\saira\\\\Downloads\\\\archive (1).zip\", encoding= 'ISO-8859-1', encoding_errors = 'strict')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "33cb6445-bb28-41f4-a4b3-68026084e338",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>v1</th>\n",
       "      <th>v2</th>\n",
       "      <th>Unnamed: 2</th>\n",
       "      <th>Unnamed: 3</th>\n",
       "      <th>Unnamed: 4</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ham</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ham</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>spam</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>ham</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ham</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     v1                                                 v2 Unnamed: 2  \\\n",
       "0   ham  Go until jurong point, crazy.. Available only ...        NaN   \n",
       "1   ham                      Ok lar... Joking wif u oni...        NaN   \n",
       "2  spam  Free entry in 2 a wkly comp to win FA Cup fina...        NaN   \n",
       "3   ham  U dun say so early hor... U c already then say...        NaN   \n",
       "4   ham  Nah I don't think he goes to usf, he lives aro...        NaN   \n",
       "\n",
       "  Unnamed: 3 Unnamed: 4  \n",
       "0        NaN        NaN  \n",
       "1        NaN        NaN  \n",
       "2        NaN        NaN  \n",
       "3        NaN        NaN  \n",
       "4        NaN        NaN  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "e59f47a9-30d1-43d7-91e6-5a08e2e0bc27",
   "metadata": {},
   "outputs": [],
   "source": [
    "# cleaning this dataset\n",
    "\n",
    "df=df.drop(df.columns[[2,3,4]], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "792c9324-2b14-4ddb-89c9-167dbcc6abad",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>v1</th>\n",
       "      <th>v2</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ham</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ham</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>spam</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>ham</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ham</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     v1                                                 v2\n",
       "0   ham  Go until jurong point, crazy.. Available only ...\n",
       "1   ham                      Ok lar... Joking wif u oni...\n",
       "2  spam  Free entry in 2 a wkly comp to win FA Cup fina...\n",
       "3   ham  U dun say so early hor... U c already then say...\n",
       "4   ham  Nah I don't think he goes to usf, he lives aro..."
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "3dda736a-53a1-46c0-bb70-2c09d227b795",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Renaming multiple columns\n",
    "df.rename(columns={'v1': 'Label', 'v2': 'Text'}, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "f0728197-3ab6-46bd-a74e-0dba0c630e77",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Label</th>\n",
       "      <th>Text</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ham</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ham</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>spam</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>ham</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ham</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Label                                               Text\n",
       "0   ham  Go until jurong point, crazy.. Available only ...\n",
       "1   ham                      Ok lar... Joking wif u oni...\n",
       "2  spam  Free entry in 2 a wkly comp to win FA Cup fina...\n",
       "3   ham  U dun say so early hor... U c already then say...\n",
       "4   ham  Nah I don't think he goes to usf, he lives aro..."
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "aba0c478-70e4-4665-a3f4-4a2d772907f2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...'"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Text[0]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "3c9561e1-75e6-4c90-8954-c0716bbd7377",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Acutally this data are contained huge ammount of space and dot. thats why we can use regex for given only words and numbers.\n",
    "# Apply the regular expression to the \"Text\" column and join the results with a space\n",
    "df['Text'] = df['Text'].apply(lambda x: ' '.join(re.findall(r'[A-Za-z0-9]+', str(x))))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "f6b10775-4875-4915-b327-d6eff4a9a903",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Go until jurong point crazy Available only in bugis n great world la e buffet Cine there got amore wat'"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Text'][0]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "d4da0a03-f975-47c7-9055-75e9cdad2f15",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Label data are convert into a decimal number just label encoding using by lambda\n",
    "df.Label=df.Label.apply(lambda x: 1 if x==\"spam\" else 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "48baf299-8fd3-4fc5-967c-1b02aa856f90",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Label</th>\n",
       "      <th>Text</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>Go until jurong point crazy Available only in ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>Ok lar Joking wif u oni</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>U dun say so early hor U c already then say</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>Nah I don t think he goes to usf he lives arou...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Label                                               Text\n",
       "0      0  Go until jurong point crazy Available only in ...\n",
       "1      0                            Ok lar Joking wif u oni\n",
       "2      1  Free entry in 2 a wkly comp to win FA Cup fina...\n",
       "3      0        U dun say so early hor U c already then say\n",
       "4      0  Nah I don t think he goes to usf he lives arou..."
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "f067e601-4e55-4c18-8405-b131dc20a1f3",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "8a674479-1b94-43b6-a9f5-af5ae5164af8",
   "metadata": {},
   "outputs": [],
   "source": [
    " #data are divied into training and testing dataset\n",
    "X_train, X_test, y_train, y_test=train_test_split(df.Text, df.Label, test_size=0.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "65cf80ed-9597-4a89-b908-2a2d75052465",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(4457,)"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.shape\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "52422227-1948-4c04-ba79-488d12e155d7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# import counvectorization for text transfrom \n",
    "from sklearn.feature_extraction.text import CountVectorizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "931dddc9-2452-4b96-a1a5-8d2238f0ed1b",
   "metadata": {},
   "outputs": [],
   "source": [
    "cv=CountVectorizer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "53963af1-ac5c-466e-bb6d-7ef58847ae2e",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_cv= cv.fit_transform(X_train.values)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "4e2ddd3e-9186-400d-8c6f-206c35aee24a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "726     Of cos can lar i m not so ba dao ok 1 pm lor Y...\n",
       "3209    She said do u mind if I go into the bedroom fo...\n",
       "3937    WHEN THE FIRST STRIKE IS A RED ONE THE BIRD AN...\n",
       "5334     Garbage bags eggs jam bread hannaford wheat chex\n",
       "5153    Haven t left yet so probably gonna be here til...\n",
       "                              ...                        \n",
       "3227    SIX chances to win CASH From 100 to 20 000 pou...\n",
       "2540    They said if its gonna snow it will start arou...\n",
       "732     Lol you won t feel bad when I use her money to...\n",
       "1557                                        Wat r u doing\n",
       "2940    My supervisor find 4 me one lor i thk his stud...\n",
       "Name: Text, Length: 4457, dtype: object"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "X_train\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "c63a4e95-0801-49ac-9007-d0540ce5209b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0, 0, 0, ..., 0, 0, 0],\n",
       "       [0, 0, 0, ..., 0, 0, 0],\n",
       "       [0, 0, 0, ..., 0, 0, 0],\n",
       "       ...,\n",
       "       [0, 0, 0, ..., 0, 0, 0],\n",
       "       [0, 0, 0, ..., 0, 0, 0],\n",
       "       [0, 0, 0, ..., 0, 0, 0]], dtype=int64)"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train_cv.toarray()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "12ea6bc3-14bd-4ffc-b037-9f8429ab46b5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(4457, 7594)"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    " # there are 7676 unique words are locate in our BOW\n",
    "X_train_cv.shape\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "f760263e-7548-4ddc-a68e-c35747b3eedc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['cost', 'costa', 'costing', 'costs', 'couch', 'cougar', 'cough',\n",
       "       'coughing', 'could', 'coulda', 'couldn', 'count', 'countin',\n",
       "       'countinlots', 'country', 'counts', 'coupla', 'couple', 'courage',\n",
       "       'course', 'courtroom', 'cousin', 'cover', 'coveragd', 'covers',\n",
       "       'coz', 'cozy', 'cps', 'cr01327bt', 'cr9', 'crack', 'craigslist',\n",
       "       'crammed', 'cramps', 'crap', 'crash', 'crashed', 'crashing',\n",
       "       'crave', 'craving', 'craziest', 'crazy', 'crazyin', 'crckt',\n",
       "       'cream', 'created', 'creative', 'creativity', 'credit', 'credited',\n",
       "       'credits', 'creepy', 'cres', 'cribbs', 'cricketer', 'crickiting',\n",
       "       'cried', 'crisis', 'cro1327', 'crore', 'cross', 'crowd', 'croydon',\n",
       "       'crucial', 'cruel', 'cruise', 'cruisin', 'crushes', 'cry', 'cs',\n",
       "       'csbcm4235wc1n3xx', 'csh11', 'cst', 'cstore', 'ctagg', 'ctargg',\n",
       "       'cthen', 'cttargg', 'ctter', 'cttergg', 'ctxt', 'cuck', 'cud',\n",
       "       'cuddle', 'cuddled', 'cuddling', 'culdnt', 'cultures', 'cum',\n",
       "       'cumming', 'cup', 'cupboard', 'cuppa', 'curfew', 'curious',\n",
       "       'current', 'currently', 'curry', 'curtsey', 'cusoon'], dtype=object)"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cv.get_feature_names_out()[2000:2100]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "26e06a1f-79e9-49cd-a228-bfb1f93144e1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, ..., 0, 0, 0], dtype=int64)"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#cv.vocabulary_\n",
    "X_train_np=X_train_cv.toarray()\n",
    "X_train_np[0]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "15c5d4da-1292-437d-b1ca-fc55677fdd7e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([ 851, 1076, 1165, 1554, 1609, 1998, 2134, 2942, 3100, 3302, 3939,\n",
       "        4117, 4658, 4732, 4794, 4817, 4836, 5157, 5762, 6146, 6807, 7273,\n",
       "        7346, 7378, 7461], dtype=int64),)"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.where(X_train_np[0]!=0)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "728279c7-f935-4a6e-b8d3-ad3d92274710",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([  34,  344,  764, 2885, 5310, 6316, 7561], dtype=int64),)"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.where(X_train_np[2961]!=0)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "62c3f59c-ef0b-470c-9594-5c97b887c8ba",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train_np[2961][1644]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "e90706f9-6944-49cf-a974-87abf41d4b9e",
   "metadata": {},
   "outputs": [],
   "source": [
    "#model /classify this model\n",
    "# impor Naive Bayes algorithm for prediction \n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "NB=MultinomialNB()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "e0cdf5ed-0c76-4edd-a2cd-6a148b5749cb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>MultinomialNB()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">MultinomialNB</label><div class=\"sk-toggleable__content\"><pre>MultinomialNB()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "MultinomialNB()"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "NB.fit(X_train_cv, y_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "1c4020d7-2026-479a-91cf-7781910b7a63",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test_cv= cv.transform(X_test)\n",
    "y_pred=NB.predict(X_test_cv)\n",
    "from sklearn.metrics import accuracy_score, classification_report\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "3f06bd93-633f-4911-acc3-82e96d3f3312",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.99      1.00      0.99       963\n",
      "           1       0.99      0.93      0.96       152\n",
      "\n",
      "    accuracy                           0.99      1115\n",
      "   macro avg       0.99      0.97      0.98      1115\n",
      "weighted avg       0.99      0.99      0.99      1115\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test, y_pred))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "bf535e43-f142-433c-90b5-8dea5713e959",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.989237668161435\n"
     ]
    }
   ],
   "source": [
    "print(accuracy_score(y_test, y_pred))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "694288cc-23b9-485c-9d29-25b12f4c918b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
