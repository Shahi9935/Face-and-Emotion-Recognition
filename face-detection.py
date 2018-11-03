import cv2
import os
import time
import numpy as np
import yaml


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    
    if (len(faces) == 0):
        return [],[]

    xyz=[]
    
    for j in faces:
        (x, y, w, h) = j
        xyz.append(gray[y:y+w, x:x+h])
    
    return xyz, faces


def detect_face_only(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    
    if (len(faces) == 0):
        return None, None
   
    (x, y, w, h) = faces[0]
    
    return gray[y:y+w, x:x+h], faces[0]


def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    model={}
    with open("data.yml", 'r') as stream:
        model = yaml.load(stream)
    label=0
    maxa=0
    for dir_name in dirs:
        label = int(dir_name.replace("s", ""))
        if(label>maxa):
            maxa=label
    label=maxa+1

    name=input("Enter your name")
    dirName='s'+str(label)
    i = 0
    cv2.namedWindow("camera", 1)
    capture = cv2.VideoCapture(0)
    path = r"C:\Users\SHAHI\Desktop\Face Detection\training-data\\"+dirName
    os.mkdir(path)
    while True:
        ret,img = capture.read()
        cv2.imshow('Video', img)
        path=r'C:\Users\SHAHI\Desktop\Face Detection\training-data\\'+dirName
        print('YES')
        cv2.imwrite(os.path.join(path , 'pic{:>05}.jpg'.format(i)), img)
        if i==200:
            break
        i+=1



    
    
    dict={}
    i=0
    dict['name']=name
    dict['value']=[]
    dict['label']=[]
    
            
    label = int(dirName.replace("s", ""))
    
    subject_dir_path = data_folder_path + "/" + dirName
    
    subject_images_names = os.listdir(subject_dir_path)
    
    for image_name in subject_images_names:
        
        if image_name.startswith("."):
            continue;
        i+=1
        image_path = subject_dir_path + "/" + image_name
        image = cv2.imread(image_path)
            
        cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
        cv2.waitKey(100)
            
        face, rect = detect_face_only(image)
            
            
        if face is not None:
            dict['value'].append(face)
            dict['label'].append(label)
            faces.append(face)
            labels.append(label)
    model['models'].append(dict)
    with open('data.yml', 'w') as outfile:
        yaml.dump(model, outfile, default_flow_style=False)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels







def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)




def predict(test_img):
    img = test_img.copy()
    faces, rects = detect_face(img)

    for face,rect in zip(faces,rects):
        if not face is None:
            label, confidence = face_recognizer.predict(face)
            label_text = subjects[label]
            
            draw_rectangle(img, rect)
            draw_text(img, label_text, rect[0], rect[1]-5)
        
    return img
    return test_img


subjects = [""]
while True:
    x=int(input("Choose you option\n1. Add an image\n2. View magic\n"))
    if(x==1):
        faces, labels = prepare_training_data("training-data")
    elif(x==2):
        faces=[]
        labels=[]
        with open("data.yml", 'r') as stream:
            model = yaml.load(stream)
            for i in model['models']:
                subjects.append(i['name'])
                for j in i['label']:
                    labels.append(j)
                for k in i['value']:
                    faces.append(k)
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()

        face_recognizer.train(faces, np.array(labels))
        print("Predicting images...")


        cascPath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascPath)

        video_capture = cv2.VideoCapture(0)

        while True:
            ret, frame = video_capture.read()
            if frame is None:
                continue
            predi = predict(frame)
            cv2.imshow('Video', predi)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
    else:
        print("Not a vald option\n")






