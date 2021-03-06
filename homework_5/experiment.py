import cv2
from train import processFiles, trainSVM
from detector import Detector
import numpy as np
# Replace these with the directories containing your
# positive and negative sample images, respectively.
pos_dir = "samples/vehicles"
neg_dir = "samples/non-vehicles"

# Replace this with the path to your test video file.
video_file = "videos/test_video.mp4"

def experiment1():
    """
    Train a classifier and run it on a video using default settings
    without saving any files to disk.
    """

    # Extract HOG features from images in the sample directories and return
    # results and parameters in a dict.
    feature_data = processFiles(pos_dir, neg_dir, recurse=True,
        hog_features=True)

    # Train SVM and return the classifier and parameters in a dict.
    # This function takes the dict from processFiles() as an input arg.
    classifier_data = trainSVM(feature_data=feature_data, output_file=True,output_filename='svm_1')
	

	##TODO: If you have trained your classifier and prepare to detect the video, 
	##      uncomment the code below.
	
    # Instantiate a Detector object and load the dict from trainSVM().
    detector = Detector().loadClassifier(classifier_data=classifier_data)

    # Open a VideoCapture object for the video file.
    cap = cv2.VideoCapture(video_file)

    # Start the detector by supplying it with the VideoCapture object.
    # At this point, the video will be displayed, with bounding boxes
    # drawn around detected objects per the method detailed in README.md.
    detector.detectVideo(video_capture=cap)

def experiment2():
    overlap_test = np.linspace(0.5,1,50)
    for i in range(len(overlap_test)):
        detector = Detector(x_overlap=overlap_test[i]).loadClassifier(filepath='svm_1')
        cap = cv2.VideoCapture(video_file)
        detector.detectVideo(video_capture=cap)

def experiment3():
    feature_data = processFiles(pos_dir, neg_dir, recurse=True,
                                hog_features=True)

    # Train SVM and return the classifier and parameters in a dict.
    # This function takes the dict from processFiles() as an input arg.
    classifier_data = trainSVM(feature_data=feature_data, output_file=True, output_filename='svm_1')
    detector = Detector(init_size=(128, 128), x_overlap=0.8, y_step=0.01, x_range=(0.4, 1),
                        y_range=(0.5, 0.8), scale=0.6).loadClassifier(filepath='svm_1')
    cap = cv2.VideoCapture(video_file)
    detector.detectVideo(video_capture=cap, write=True, num_frames=4, threshold=45)
#def experiment2
#    ...

if __name__ == "__main__":
    experiment3()
    #experiment2() may you need to try other parameters
	#experiment3 ...