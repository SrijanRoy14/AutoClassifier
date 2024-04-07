import tensorflow as tf
import streamlit as st
import os
import shutil
import hashlib
import random
import cv2
import imghdr
import keras
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from PIL import Image


#from keras.engine.sequential import Sequential
from keras import Sequential

from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential


def check_gpu(self):
    gpus=tf.config.experimental.list_physical_devices('GPU')#selects the available gpus in the device
    if(gpus is not None):
        for gpu in gpus:
            #print(gpus)
            tf.config.experimental.set_memory_growth(gpu,True)#tells tf to minimise the gpu usage, sets a limit
        return True
    else:
        return False

def remove_directories():
    if(os.path.isdir('sessions')==True):shutil.rmtree('sessions')
    

def reset_session(flag):
    #shutil.rmtree('sessions')
    remove_directories()
    #st.write("resetting sessions")
    st.cache_data.clear()
    st.cache_resource.clear()
    for key in st.session_state.keys():
        del st.session_state[key]
    if flag==1:
        st.rerun()


def calculate_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


@st.cache_data
def find_duplicate_files(root_folder):
    
    duplicates = {}
    for folder_path, _, file_names in os.walk(root_folder):
        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)
            file_hash = calculate_file_hash(file_path)
            if file_hash in duplicates:
                duplicates[file_hash].append(file_path)
            else:
                duplicates[file_hash] = [file_path]
    return duplicates



@st.cache_data
def remove_duplicate_files(duplicates,cl):
    
    check=False
    for file_paths in duplicates.values():
        if len(file_paths) > 1:
            print(f"Duplicate files found:\n{file_paths}\n")
            for file_path in file_paths[1:]:
                os.remove(file_path)
                st.error(f"###### {file_path} was found to be a duplicate and has been deleted.\n")
    if(check==False):
        st.success("###### No duplicate images were found for class {}!".format(cl))


@st.cache_data(show_spinner=False)
def download_images(q,cl,n):
    if(os.path.isdir('./sessions/data/train/'+cl)==True):shutil.rmtree('./sessions/data/train/'+cl)
    os.system('bbid.py {} --limit {} -o "./sessions/data/train/{}/" --filter +filterui:imagesize-large+filterui:photo-photo'.format(q,n,cl))
    class_size=len(os.listdir('sessions/data/train/'+cl))
    return class_size
        

@st.cache_data      
def imbalance(dif,cl):
    img_list=os.listdir('sessions/data/train/'+cl)
    for i in range (0,dif):
        os.remove('sessions/data/train/'+cl+'/'+random.choice(img_list))


@st.cache_data(show_spinner=False)
def dodgy(data_dir):
    image_exts= ['jpeg','jpg','bmp','png']
    counter=0
    for image_class in os.listdir(data_dir):#gives the classes
        for image in os.listdir(os.path.join(data_dir,image_class)):#gives the path of each image class
            image_path=os.path.join(data_dir,image_class,image)#gives the image path
            try:
                img=cv2.imread(image_path)#load the image in a numpy array
                tip=imghdr.what(image_path)#checks the extension
                if tip not in image_exts:
                    print("Image not in ext list {} removing it!".format(image_path))
                    os.remove(image_path)
                    counter+=1
            except Exception as e:
                print(e)
                print('Issue with image {} removing it!'.format(image_path))
                counter+=1
    return counter


@st.cache_data
def display_image(root,cl):
    img_list=os.listdir(root+cl)
    img=[]
    for i in range (0,4):
        img.append(root+cl+"/"+img_list[i])
    #st.write(img)
    row=st.columns(4)
    for col,i in zip(row,img):
        tile=col.container(height=260,border=False)
        image = Image.open(i)
        resized_image = image.resize((260,260))
        tile.image(resized_image,caption=cl)

    

@st.cache_resource
def data_pipeline():
    train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
    return train_generator,validation_generator


@st.cache_resource(show_spinner=False)
def build_model(input_shape,output_shape):
    base_model = keras.applications.inception_v3.InceptionV3(
					include_top=False,
					weights='imagenet',
					input_shape=input_shape
					)

    base_model.trainable=False

    model = keras.Sequential([
            base_model,
            keras.layers.BatchNormalization(renorm=True),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(output_shape, activation='softmax')
        ])
    
    early = tf.keras.callbacks.EarlyStopping( patience=4,min_delta=0.001,restore_best_weights=True,monitor='val_loss')
    log=tf.keras.callbacks.TensorBoard(log_dir='sessions/logs/fit',histogram_freq=1)

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    call_backs=[early,log]
    return model,call_backs


@st.cache_resource(show_spinner=False)
def fit_model(epochs):
    history=model.fit(
    train_generator,
    steps_per_epoch=nb_train_sample // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=nb_validation_sample // batch_size,
    callbacks=call_backs)
    return model,history


@st.cache_data
def make_validation(cl):
    if(os.path.isdir('./sessions/data/val/'+cl)==True):shutil.rmtree('./sessions/data/val/'+cl)

    img_list=os.listdir('sessions/data/train/'+cl)
    size=int(len(img_list)*0.2)
    
    if(os.path.isdir('sessions/data/val/'+cl)==False):os.makedirs('sessions/data/val/'+cl)
    for i in range(0,size+1):
        
        val_path=os.path.join('sessions/data/val/'+cl)
        shutil.move('sessions/data/train/'+cl+'/'+img_list[i],val_path)


def visualise(history):
    viz_df=pd.DataFrame(history.history,index=[i for i in range (1,len(history.history['val_loss'])+1)])
    viz_df['epoch']=[i for i  in range(1,len(history.history['val_loss'])+1)]

    #viz_df.set_index('epoch',inplace=True)
    tab1,tab2,tab3=st.tabs(["📈 Accuracy Chart", "📈 Loss Chart","🗃 Data"])
    
    tab1.subheader("Accuracy of your model")
    tab1.line_chart(viz_df,x="epoch",y=["val_accuracy","accuracy"])
    
    tab2.subheader("Loss of your model")
    tab2.line_chart(viz_df,x="epoch",y=["val_loss","loss"])

    tab3.subheader("The data looks like")
    tab3.table(viz_df)
    



def make_prediction(model,file,col2):
    image=Image.open(file).convert('RGB')
    col2.image(image,width=300)
    resize=tf.image.resize(np.array(image),(150,150))
    pred=model.predict(np.expand_dims(resize/255,0))
    return pred


def metrices(pred,col2):
    m=np.argmax(pred)
    conf=pred[0][m]*100
    predicted_class_indices=np.argmax(pred,axis=1)
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]
    col2.success("predicted {} with a confidence of {:.2f} %".format(predictions[0],conf))



@st.cache_resource(hash_funcs={Sequential: lambda x: x.name})
def save_model(model):
    save_dir="sessions/models/generated_model{}and{}Classifier.h5".format(classes[0],classes[1])
    tf.keras.saving.save_model(model,save_dir)
    return save_dir



def down_model(save_dir,col1):
    with open(save_dir, "rb") as file:
        btn = col1.download_button(
                label="Download model",
                data=file,
                file_name="trainedmodel.h5",)



def remove_extra(path,classes):
    remove=[]
    for li in os.listdir(path):
        if(li not in classes):
            remove.append(li)
    if len(remove)>0:
        st.write(remove)
        for r in remove:
            shutil.rmtree(path+'/'+r)


def set_state(i):
    st.session_state.stage = i
def increase_rows():
    st.session_state['rows'] += 1
def display_input_row(index):
    q, cl = st.columns(spec=[0.7,0.3])
    q.text_input('Enter Query', key=f'query_{index}')
    if((index-1))>=0:
        if(str(st.session_state[f'query_{index}'])==str(st.session_state[f'query_{(index-1)}'])):
            st.warning("Same input as previous quary")
    if(st.session_state[f'query_{index}']!=''):
        cl.text_input('Enter name for class', key=f'class_{index}',on_change=show_button,kwargs=dict(index=index))
def show_button(index):
    col1,col2=st.columns(spec=[0.75,0.2],gap="large")
    col1.button('Add a query', on_click=increase_rows)
    if(index>=1):
        col2.button('Download', on_click=set_state,args=[2],type="primary")
       



root_folder = 'sessions/data/train/'
if 'class_sizes' not in st.session_state:
    st.session_state.class_sizes = {}
if 'val_sizes' not in st.session_state:
    st.session_state.val_sizes = {}
if 'n_imgs' not in st.session_state:
    st.session_state['n_imgs'] = 0
if 'stage' not in st.session_state:
    st.session_state.stage = 1
if 'quer' not in st.session_state:
    st.session_state['quer']=[]
if 'class' not in st.session_state:
    st.session_state['class']=[]
if 'rows' not in st.session_state:
    st.session_state['rows'] = 1


st.title("Auto Classifier")


if st.session_state.stage == 1:
    st.subheader("*Section: 1*")
    st.write("###### This is the data ingestion stage. Two steps will be performed: ")
    st.write("- A specified number of images will be downloaded from the web")
    st.write("- Visualisation of the downloaded images!")
    for i in range(st.session_state['rows']):
        display_input_row(i)



if st.session_state.stage == 2:   
    for i in range(st.session_state['rows']):
        st.session_state['class'].append(st.session_state[f'class_{i}'])
        st.session_state['quer'].append(st.session_state[f'query_{i}'])
    set_state(3)



if st.session_state.stage == 3:
    
    query=st.session_state['quer']
    classes=st.session_state['class']
    #st.write(query,classes)
    
                
    #download
    #remove_directories()
    st.session_state['n_imgs']=st.number_input('Enter the no. of images to download',10,300)
    if st.session_state['n_imgs']>10:
        for q,cl in zip(query,classes):
            with st.spinner("*Please wait...downloading {} images for class {}*".format(st.session_state['n_imgs'],cl)):
                class_size=download_images(q,cl,st.session_state['n_imgs'])
                st.session_state.class_sizes[cl]=class_size
            
        
        st.table(pd.DataFrame({'Class':st.session_state.class_sizes.keys(),'No. of images':st.session_state.class_sizes.values()},index=[i for i in range(1,len(st.session_state.class_sizes.keys())+1)]))
        
        st.success("Sweet now lets visualise some of the downloaded images!")       
        st.button('Visualisation', on_click=set_state,args=[5])

        

if st.session_state.stage == 5:
    
    query=st.session_state['quer']
    classes=st.session_state['class']  
    
    for cl in classes:
        make_validation(cl)
        
    for  cl in classes:
        st.write("##### Lets visualise the downloaded images for class {}".format(cl))
        display_image(root_folder,cl)

    col1,col2=st.columns(spec=[0.6,0.4])
    col1.success("Happy with the results lets proceed to preprocessing!")
    
    col2.error("Reset and redownload images! ")

    col1,col2=st.columns(spec=[0.8,0.2],gap="large")
    col1.button('Preprocessing', on_click=set_state,args=[6],type="secondary")
    col2.button('Reset', on_click=reset_session,args=[1],type="primary")

    
if st.session_state.stage==6:
    
    st.subheader("*Section: 2*")
    st.write("##### This is the data preprocessing stage. There will be two steps:")
    st.write("- Removing duplicate images")
    st.write("- Removing dodgy images")
    query=st.session_state['quer']
    classes=st.session_state['class']  
    try:
        #duplicate
        for cl in classes:
            with st.spinner("please wait finding and removing duplicate images for class {}".format(cl)):
                duplicates = find_duplicate_files(root_folder+cl)
                remove_duplicate_files(duplicates,cl)
        
        #dodgy
        with st.spinner("please wait finding and removing dodgy images for {} classes".format(len(classes))):
            counter=dodgy(root_folder)
        if counter!=0:
            st.error("{} dodgy images were found and removed!".format(counter))
        else:
            st.success("{} dodgy images were found!".format(counter))
        

        for cl in classes:
            class_size=len(os.listdir('sessions/data/train/'+cl))
            val_size=len(os.listdir('sessions/data/val/'+cl))
            st.session_state.class_sizes[cl]=class_size
            st.session_state.val_sizes[cl]=val_size

        st.write("###### Final number of train and validation images: ") 

        st.table(pd.DataFrame({'Class Name':st.session_state.class_sizes.keys(),'No. of train images':st.session_state.class_sizes.values(),'No. of val images':st.session_state.val_sizes.values()},index=[i for i in range(1,len(st.session_state.class_sizes.keys())+1)]))

        st.write("###### Now lets move on to the third section the model training!")

        if len(classes)<len(os.listdir("./sessions/data/train")) or len(classes)<len(os.listdir("./sessions/data/val")):
            remove_extra("./sessions/data/train",classes)
            remove_extra("./sessions/data/val",classes)

        col1,col2=st.columns(spec=[0.8,0.2],gap="large")
        col1.button('Model Training', on_click=set_state,args=[7],type="secondary")
        col2.button('Reset', on_click=reset_session,args=[1],type="primary")
    except Exception as e:
        st.error(e)
        st.error("please reset the app")
        st.button('reset', on_click=reset_session,args=[1],type="primary")


    


if st.session_state.stage==7: 
    
    st.subheader("*Section: 3*")
    st.write("##### This is the Model building stage. There will be three step:")
    st.write("- Model training")
    st.write("- Making prediction") 
    st.write("- Finally downloading your trained model!") 
    
    query=st.session_state['quer']
    classes=st.session_state['class']
    
    st.session_state['epoch']=st.number_input("Enter the number of epochs",1,75)
    
    if st.session_state['epoch']>1:
        
        #the data pipeline
        img_width, img_height = 150, 150
        train_data_dir = 'sessions/data/train'
        validation_data_dir = 'sessions/data/val'
        epochs = st.session_state['epoch']
        batch_size = 2
        nb_train_sample=0
        nb_validation_sample=0
        
        try:
            for cl in classes:
                nb_train_sample =nb_train_sample+len(os.listdir('sessions/data/train/'+cl))
                nb_validation_sample =nb_validation_sample+len(os.listdir('sessions/data/val/'+cl))
        except Exception as e:
            st.error(e)
            st.error("Please reset and rerun")
            st.button('reset', on_click=reset_session,args=[1],type="primary")

        
        train_generator,val_generator=data_pipeline()

        #build model
        with st.spinner("Building your model"):
            model,call_backs=build_model((150,150,3),len(classes))

        #fit model
        try:
            with st.spinner("Model training in progress...this might take a while"):
                model,history=fit_model(epochs)
            st.success("Sweet your model is trained now!")
        except Exception as e:
            st.error(e)
            st.error("Please reset and rerun")
            st.button('reset', on_click=reset_session,args=[1],type="primary")

        with st.expander("##### Visualise the performance of your model"):

            #viualise the data
            visualise(history)

        st.write("##### Lets make some predictions!")
        col1,col2,col3=st.columns([0.1,0.8,0.2],gap="medium")

        
        #make pred
        file=col2.file_uploader('',type=['jpeg','jpg','bmp','png'])
        try:
            if file is not None:
                
                pred=make_prediction(model,file,col2)
        
                #pred_metrices
                metrices(pred,col2)

                #download
                save_dir=save_model(model)

                st.write("##### Happy with the results? You can also download your model!")

                col1,col2=st.columns(spec=[0.8,0.2],gap="large")
               
                #down
                down_model(save_dir,col1)

                col2.button('Reset',on_click=reset_session,args=[1],type="primary")
        except Exception as e:
            st.error(e)
            st.button('reset', on_click=reset_session,args=[1],type="primary")
