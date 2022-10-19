# CBML - ChatBotMachineLite
# Refer to LICENSE for more information
# (c) Rizaldy Aristyo 2022 Licensed under Apache 2.0 License (See LICENSE)

import os,random,pickle,json
from collections import OrderedDict
try:import nltk
except:raise ImportError("Please install nltk module")
try:import numpy
except:raise ImportError("Please install numpy module")
try:from nltk.stem import WordNetLemmatizer
except:raise ImportError("Please install nltk\nDon't forget to run\n  python -c import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')\nafter installing nltk")
lemmatizer=WordNetLemmatizer()

def replaceall(userinput:str,replacements_dict:OrderedDict) -> str or bool:
    """
    Required by respond() and respond_json()
    Replaces all the keys in the replacements_dict with the values
    So you can hold variables in your json corpus

    Example:
        STRING_REPLACEMENTS = OrderedDict([
            ("$MACHINE_NAME","Arisha"),
            ("$MACHINE_VERSION","1.0"),
            ("$MACHINE_AGE","21"),
        ])
        cbml.respond("What's Your Name?",STRING_REPLACEMENTS)
        original response: ----> My name is $MACHINE_NAME
        replaced response: ----> My name is Arisha
    """
    try:
        for s,r in replacements_dict.items():
            userinput=userinput.replace(s,r)
        return userinput
    except:
        return False

def load(json_corpus:str,h5_model:str="model.h5",pickle_words:str="words_corpuspickle.pkl",pickle_classes:str="classes_corpuspickle.pkl",verbosity:int=0)->bool:
    """
    Loads the model, corpus, words pickle, and classes pickle
    Returns True if successful, False if not
    You can use the default values or specify your own
    and you can also enable the verbosity to see the loading process
    verbosity level varies from 0 to 2

    Example:
        - cbml.load("corpus.json")
        - cbml.load("corpus.json",1)
        - cbml.load("corpus.json","model.h5","words_corpuspickle.pkl","classes_corpuspickle.pkl",2)
    """
    try:
        if verbosity>2:
            return False
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if verbosity == 2 else '3' 
        print("\nLoading Model and Corpus..." if verbosity==1 or verbosity==2 else "",end="")
        from keras.models import load_model
        global model,intents,words,classes
        model=load_model(h5_model)
        intents=json.loads(open(json_corpus).read())
        words=pickle.load(open(pickle_words,'rb'))
        classes=pickle.load(open(pickle_classes,'rb'))
        print("\nLoading Complete" if verbosity==1 or verbosity==2 else "",end="")
        return True
    except Exception as e:
        return False

def train(json_corpus:str,save_model_as:str="model.h5",save_pickle_as:str="corpuspickle.pkl",epoch_value:int=10,batch_size_value:int=128,verbosity:int=0)->bool:
    """
    Trains a model based on the json corpus
    Returns True if successful, False if not
    You can use the default values or specify your own
    and you can also enable the verbosity to see the loading process
    verbosity level varies from 0 to 2

    Example:
        - cbml.load("corpus.json")
        - cbml.load("corpus.json",1)
        - cbml.load("corpus.json","model.h5","words_corpuspickle.pkl","classes_corpuspickle.pkl",10,128,2)
    """
    try:
        if verbosity>2:
            return False
        if verbosity == 2:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        else:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        from tensorflow.keras.layers import Flatten,Input,Dense,Dropout
        from tensorflow.keras.models import Model
        print("\nPreparing Corpus..." if verbosity==1 or verbosity==2 else "",end="")
        documents=[]
        words=[]
        classes=[]
        ignorethese=['?','!']
        intents=json.loads(open(json_corpus).read())
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                w=nltk.word_tokenize(pattern)
                words.extend(w)
                documents.append((w,intent['class']))
                if intent['class'] not in classes:
                    classes.append(intent['class'])
        words=[lemmatizer.lemmatize(w.lower()) for w in words if w not in ignorethese]
        words=sorted(list(set(words)))
        classes=sorted(list(set(classes)))
        pickle.dump(words,open('words_'+save_pickle_as,'wb+'))
        pickle.dump(classes,open('classes_'+save_pickle_as,'wb+'))
        training=[]
        output_empty=[0]*len(classes)
        for doc in documents:
            bag=[]
            pattern_words=doc[0]
            pattern_words=[lemmatizer.lemmatize(word.lower()) for word in pattern_words]
            for w in words:
                bag.append(1) if w in pattern_words else bag.append(0)
                output_row=list(output_empty)
                output_row[classes.index(doc[1])]=1
                training.append([bag,output_row])
        random.shuffle(training)
        training=numpy.array(training, dtype=object)
        train_x=list(training[:,0])
        train_y=list(training[:,1])
        input_layer=Input(shape=(len(train_x[0]),))
        print("\nBuilding Model..." if verbosity==1 or verbosity==2 else "",end="")
        x = Flatten()(input_layer)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        output_layer=Dense(units=len(train_y[0]), activation='softmax')(x)
        model = Model(inputs=input_layer, outputs=output_layer)
        print("\nCompiling Model..." if verbosity==1 or verbosity==2 else "",end="")
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        print("\nTraining on "+str(epoch_value)+" epochs and "+str(batch_size_value)+" batch size\n" if verbosity==1 or verbosity==2 else "",end="")
        hist=model.fit(numpy.array(train_x),numpy.array(train_y),epochs=epoch_value,batch_size=batch_size_value,verbose=1 if verbosity==1 or verbosity==2 else 0)
        print("\nPreparing Corpus..." if verbosity==1 or verbosity==2 else "",end="")
        model.save(save_model_as,hist)
        return True
    except Exception as e:
        return False

def train_and_load(json_corpus:str,save_model_as:str="model.h5",save_pickle_as:str="corpuspickle.pkl",epoch_value:int=10,batch_size_value:int=128,verbosity:int=0)->bool:
    """
    It does the same thing as train() and load() but in one function
    Returns True if successful, False if not
    You can use the default values or specify your own
    and you can also enable the verbosity to see the loading process
    verbosity level varies from 0 to 2

    Example:
        - cbml.train_and_load("corpus.json")
    """
    try:
        if verbosity>2:
            return False
        train(json_corpus,save_model_as,save_pickle_as,epoch_value,batch_size_value,verbosity)
        issuccess=load(json_corpus,save_model_as,'words_'+save_pickle_as,'classes_'+save_pickle_as,verbosity)
        print("\nLoading Returns: "+str(issuccess)+"\n" if verbosity==1 or verbosity==2 else "",end="")
        return issuccess
    except Exception as e:
        return False

def respond(userinput:str,undefined_response:str="I'm sorry, I don't understand",ambiguous_suffix:str="...?",replacement_ordereddict:OrderedDict=OrderedDict([]),verbosity:int=0)->str or bool:
    try:
        if verbosity>2:
            return False
        if userinput.lower()=='' or userinput.lower()== '*':
            return undefined_response
        else:
            sentence_words = nltk.word_tokenize(userinput)
            sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
            bag=[0]*len(words) 
            for s in sentence_words:
                for i,w in enumerate(words):
                    if w==s:
                        bag[i]=1
            p=numpy.array(bag)
            results=[[i,r] for i,r in enumerate(model.predict(numpy.array([p]),verbose=1 if verbosity==1 or verbosity==2 else 0)[0]) if r>0.25]
            results.sort(key=lambda x: x[1], reverse=True)
            return_list = []
            for r in results:
                return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
            print(("\nIntent: ", return_list[0]['intent']) if verbosity==1 or verbosity==2 else "",end="")
            print(("Confidence: ",float(return_list[0]['probability'])*100,"%") if verbosity==1 or verbosity==2 else "",end="")
            if float(return_list[0]['probability']) < 0.65:
                return undefined_response
            cls=return_list[0]['intent']
            list_of_intents = intents['intents']
            for i in list_of_intents:
                if i['class']== cls:
                    result=replaceall(random.choice(i['responses']),replacement_ordereddict)
                    break
            if float(return_list[0]['probability']) < 0.90:
                return result+ambiguous_suffix
            else:
                return result
    except Exception as e:
        return False

def respond_json(userinput:str,replacement_ordereddict:OrderedDict=OrderedDict([]),verbosity:int=0)->str or bool:
    try:
        if verbosity>2:
            return "{\"response\":\"Invalid Argument\"}"
        if userinput.lower()=='' or userinput.lower()== '*':
            return "{\"response\":\"Undefined Response\"}"
        else:
            sentence_words = nltk.word_tokenize(userinput)
            sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
            bag=[0]*len(words) 
            for s in sentence_words:
                for i,w in enumerate(words):
                    if w==s:
                        bag[i]=1
            p=numpy.array(bag)
            results=[[i,r] for i,r in enumerate(model.predict(numpy.array([p]),verbose=1 if verbosity==1 or verbosity==2 else 0)[0]) if r>0.25]
            results.sort(key=lambda x: x[1], reverse=True)
            return_list = []
            for r in results:
                return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
            print(("\nIntent: ", return_list[0]['intent']) if verbosity==1 or verbosity==2 else "",end="")
            print(("Confidence: ",float(return_list[0]['probability'])*100,"%") if verbosity==1 or verbosity==2 else "",end="")
            cls=return_list[0]['intent']
            list_of_intents = intents['intents']
            for i in list_of_intents:
                if i['class']== cls:
                    result=replaceall(random.choice(i['responses']),replacement_ordereddict)
                    break
            return "{\"response\":\""+result+"\",\"intent\":\""+str(return_list[0]['intent'])+"\",\"confidence\":\""+str(return_list[0]['probability'])+"\"}"
    except Exception as e:
        return "{\"response\":\""+str(e)+"\"}"
