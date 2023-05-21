#importing all relevant modules
import cv2
import os 
import tensorflow as tf 
import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file

#fixing seed to get reproduceable results
np.random.seed(123)

#creating our Object_Detection class

class Object_Detector:
    def __init__(self):
        pass

    #creating a function to read our coco.names class file

    def class_read(self,PathForClasses):
        with open(PathForClasses, 'r') as f:
            self.classeslist = f.read().splitlines()

        #creating a unique colour for every class label

        self.colourlist = np.random.uniform(low=0, high = 255, size=(len(self.classeslist),3))
        print(len(self.classeslist,),len(self.colourlist))
    
    #Taking ModelURL as an arguement, extracting filename from it

    def model_download(self,ModelURL):

        filename = os.path.basename(ModelURL)
        self.Name_model = filename[:filename.index('.')]
        
        #creating a cache dir for all models to be stored

        self.cacheDir = r'C:\Users\Miraj\Desktop\TENSORFLOW_OBJECT_DETECTION\pretrained_models'

        os.makedirs(self.cacheDir,exist_ok=True)

        get_file(fname=filename,
                 origin=ModelURL,
                 cache_dir=self.cacheDir,
                 cache_subdir='checkpoints',
                 extract = True)

    #Loading model using tensorflow

    def Model_load(self):
        print('Loading model'+ self.Name_model)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir,'checkpoints',self.Name_model, 'saved_model'))

        print('Model'+ self.Name_model + 'Successfully loaded')

    #creating our bounding boxes 

    def bounding_boxes(self,image,threshold = 0.5):
        
        #converting to rgb format from bgr format
        tensor_input = cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB)
        tensor_input = tf.convert_to_tensor(tensor_input,dtype=tf.uint8)
        tensor_input = tensor_input[tf.newaxis]

        #passing input tensor to model

        detections = self.model(tensor_input)
        bbxs = detections['detection_boxes'][0].numpy()
        index_classes = detections['detection_classes'][0].numpy().astype(np.int32)

        #extract confidence scores
        classscores = detections['detection_scores'][0].numpy()

        #calculating the location of bounding boxes

        imH, imW, imC = image.shape

        #using non maximum supression to get reduce overlapping of bounding boxes 
        #nbbx for non bounding box

        nbbx = tf.image.non_max_suppression(bbxs,classscores,
        max_output_size=10,
        iou_threshold= threshold,
        score_threshold=threshold)
        
        print(nbbx)

        # where bbx is for bounding box
        if len(nbbx) != 0:
            for j in nbbx:
                bbx = tuple(bbxs[j].tolist())
                classconfidence = round(100*classscores[j])
                index_class = index_classes[j]
                classlabeltext = self.classeslist[index_class].upper()
                classcolor = self.colourlist[index_class]

                #using f strings or string formatting to display bbx text

                display_text = ' {}: {}%'.format(classlabeltext,classconfidence)

                #horizontal and vertical components
                xmin, xmax, ymin, ymax = bbx
                
                #getting pixel locations
            
                xmin, xmax, ymin, ymax = (xmin*imW,xmax*imW,ymin*imH,ymax*imH)
                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

                cv2.rectangle(image, (xmin,ymin), (xmax, ymax), color = classcolor, thickness = 1)

                #creating text for labels and confidence text
                cv2.putText(image,display_text,(xmin,ymin - 10), cv2.FONT_HERSHEY_PLAIN,1,classcolor,2)

                #adding visual features for labels and confidence text 
                linewidth = min(int((xmax-xmin) * 0.2), int((ymax-ymin) * 0.2))
                cv2.line(image, (xmin,ymin), (xmin + linewidth, ymin), classcolor, thickness=2)
                cv2.line(image, (xmin,ymin), (xmin, ymin + linewidth), classcolor, thickness=2)

                cv2.line(image, (xmax,ymin), (xmax - linewidth, ymin), classcolor, thickness=3)
                cv2.line(image, (xmax,ymin), (xmax, ymin + linewidth,), classcolor, thickness=3)
                

                cv2.line(image, (xmin,ymax), (xmin + linewidth, ymax), classcolor, thickness=3)
                cv2.line(image, (xmin,ymax), (xmin, ymax - linewidth), classcolor, thickness=3)

                cv2.line(image, (xmax,ymax), (xmax - linewidth, ymax), classcolor, thickness=3)
                cv2.line(image, (xmax,ymax), (xmax, ymax - linewidth), classcolor, thickness=3)

        return image 
    


    #creating our function to detect images

    def image_prediction(self, imagepath, threshold = 0.5):
        image = cv2.imread(imagepath)
        self.bounding_boxes(image,threshold)

        cv2.imshow('image detection',image)
        cv2.waitKey(0)

    #creating our function to detect vidoes
    
    def video_prediction(self,videopath,threshold = 0.5):
        capture = cv2.VideoCapture(videopath)

        if (capture.isOpened()== False):
            print('An error occured whilst trying to open the file')
            return
        
        (true,image) = capture.read()

        while true:

            bboximage = self.bounding_boxes(image,threshold)
            cv2.imshow('video_detection',bboximage)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('e'):
                break

            (true, image) = capture.read()


        cv2.destroyAllWindows()




