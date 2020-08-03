##  This project work uses the dlib library and openCV
from __future__ import with_statement
from __future__ import absolute_import
from __future__ import division
from threading import Thread
from naoqi import ALProxy
from collections import Counter

###Specifing Credentials
from io import open
import os
##Make sure to change the path on your computer
os.environ[u"GOOGLE_APPLICATION_CREDENTIALS"]=u"C:\\Users\\eriol\\Desktop\\Package\\Botnica-10c05689f04b.json"



from six.moves import queue
from statistics import mode
from statistics import mode
from google.cloud import speech
from google.cloud import speech_v1p1beta1
from google.cloud.speech_v1p1beta1 import enums
from google.cloud.speech_v1p1beta1 import types
from Casscade_classes.haar_cascade import haarCascade
from Casscade_classes.face_landmark_detection import faceLandmarkDetection


import numpy
import cv2
import statistics
import random
import pyaudio
import datetime
import time
import re
import sys

#If True enables the verbose mode
DEBUG = True 

#Antropometric constant values of the human head. 
#Found on wikipedia and on:
# "Head-and-Face Anthropometric Survey of U.S. Respirator Users"
#
#X-Y-Z with X pointing forward and Y on the left.
#The X-Y-Z coordinates used are like the standard
# coordinates of ROS (robotic operative system)
P3D_RIGHT_SIDE = numpy.float32([-100.0, -77.5, -5.0]) #0
P3D_GONION_RIGHT = numpy.float32([-110.0, -77.5, -85.0]) #4
P3D_MENTON = numpy.float32([0.0, 0.0, -122.7]) #8
P3D_GONION_LEFT = numpy.float32([-110.0, 77.5, -85.0]) #12
P3D_LEFT_SIDE = numpy.float32([-100.0, 77.5, -5.0]) #16
P3D_FRONTAL_BREADTH_RIGHT = numpy.float32([-20.0, -56.1, 10.0]) #17
P3D_FRONTAL_BREADTH_LEFT = numpy.float32([-20.0, 56.1, 10.0]) #26
P3D_SELLION = numpy.float32([0.0, 0.0, 0.0]) #27
P3D_NOSE = numpy.float32([21.1, 0.0, -48.0]) #30
P3D_SUB_NOSE = numpy.float32([5.0, 0.0, -52.0]) #33
P3D_RIGHT_EYE = numpy.float32([-20.0, -65.5,-5.0]) #36
P3D_RIGHT_TEAR = numpy.float32([-10.0, -40.5,-5.0]) #39
P3D_LEFT_TEAR = numpy.float32([-10.0, 40.5,-5.0]) #42
P3D_LEFT_EYE = numpy.float32([-20.0, 65.5,-5.0]) #45
#P3D_LIP_RIGHT = numpy.float32([-20.0, 65.5,-5.0]) #48
#P3D_LIP_LEFT = numpy.float32([-20.0, 65.5,-5.0]) #54
P3D_STOMION = numpy.float32([10.0, 0.0, -75.0]) #62

#The points to track
#These points are the ones used by PnP
# to estimate the 3D pose of the face
TRACKED_POINTS = (0, 4, 8, 12, 16, 17, 26, 27, 30, 33, 36, 39, 42, 45, 62)
ALL_POINTS = range(0,68) #Used for debug only

# Cheat attribution parameters
cheating = False
potential_cheating = False
number_of_seconds_cheating = 0
estimated_head_left = 30
estimated_head_right = 30
switch =0
perecieve_left = 190
perecieve_right = 394

# Audio recording parameters
STREAMING_LIMIT = 240000  # 4 minutes
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
speaker_number = 0
speaker_array = []

RED = u'\033[0;31m'
GREEN = u'\033[0;32m'
YELLOW = u'\033[0;33m'

cheating_phrases_list = [u'tell', u'answer', u'question', u'what is', u'what']

reactive = True
#Nao Robot Parameters
ip_addr = "169.254.250.201"
port_num = 9559


##***************************************************************************************##
##**************************** HeadPose Estimation Code *********************************##
##***************************************************************************************##

def main():
    global potential_cheating 
    global cheating
    global number_of_seconds_cheating

    f = open(u"HeadPose_log.txt", u"a")
    f.write(u'\n##**************************** HeadPose Estimation (Group 1) *********************************##\n\n')
    f.close()

    
    #Defining the video capture object
    video_capture = cv2.VideoCapture(0)

    if(video_capture.isOpened() == False):
        f = open(u"HeadPose_log.txt", u"a")
        f.write(u'Error: the resource is busy or unvailable\n')
        f.close()
        print u"Error: the resource is busy or unvailable"
    else:
        f = open(u"HeadPose_log.txt", u"a")
        f.write(u'The video source has been opened correctly...\n')
        f.close()
        print u"The video source has been opened correctly..."

    #Create the main window and move it
    cv2.namedWindow(u'Video')
    cv2.moveWindow(u'Video', 20, 20)

    #Obtaining the CAM dimension
    cam_w = int(video_capture.get(3))
    cam_h = int(video_capture.get(4))

    size = (cam_w, cam_h)

##    Below VideoWriter object will create a frame of above defined.
##    The output is stored in 'filename.avi' file. 
    writer = cv2.VideoWriter(u'video_nao_recording.mp4',  
                         cv2.VideoWriter_fourcc(*u'DIVX'), 
                         20, size)

    #Defining the camera matrix.
    #To have better result it is necessary to find the focal
    # lenght of the camera. fx/fy are the focal lengths (in pixels) 
    # and cx/cy are the optical centres. These values can be obtained 
    # roughly by approximation, for example in a 640x480 camera:
    # cx = 640/2 = 320
    # cy = 480/2 = 240
    # fx = fy = cx/tan(60/2 * pi / 180) = 554.26
    c_x = cam_w / 2
    c_y = cam_h / 2
    f_x = c_x / numpy.tan(60/2 * numpy.pi / 180)
    f_y = f_x

    #Estimated camera matrix values.
    camera_matrix = numpy.float32([[f_x, 0.0, c_x],
                                   [0.0, f_y, c_y], 
                                   [0.0, 0.0, 1.0] ])

    print u"Estimated camera matrix: \n" + unicode(camera_matrix) + u"\n"

    #These are the camera matrix values estimated on my webcam with
    # the calibration code (see: src/calibration):
    camera_matrix = numpy.float32([[602.10618226,          0.0, 320.27333589],
                                   [         0.0, 603.55869786,  229.7537026], 
                                   [         0.0,          0.0,          1.0] ])

    #Distortion coefficients
    #camera_distortion = numpy.float32([0.0, 0.0, 0.0, 0.0, 0.0])

    #Distortion coefficients estimated by calibration
    camera_distortion = numpy.float32([ 0.06232237, -0.41559805,  0.00125389, -0.00402566,  0.04879263])


    #This matrix contains the 3D points of the
    # 11 landmarks we want to find. It has been
    # obtained from antrophometric measurement
    # on the human head.-
    landmarks_3D = numpy.float32([P3D_RIGHT_SIDE,
                                  P3D_GONION_RIGHT,
                                  P3D_MENTON,
                                  P3D_GONION_LEFT,
                                  P3D_LEFT_SIDE,
                                  P3D_FRONTAL_BREADTH_RIGHT,
                                  P3D_FRONTAL_BREADTH_LEFT,
                                  P3D_SELLION,
                                  P3D_NOSE,
                                  P3D_SUB_NOSE,
                                  P3D_RIGHT_EYE,
                                  P3D_RIGHT_TEAR,
                                  P3D_LEFT_TEAR,
                                  P3D_LEFT_EYE,
                                  P3D_STOMION])

    #Declaring the two classifiers
    my_cascade = haarCascade("etc/xml/haarcascade_frontalface_alt.xml", u"etc/xml/haarcascade_profileface.xml")
    #TODO If missing, example file can be retrieved from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    my_detector = faceLandmarkDetection('etc/shape_predictor_68_face_landmarks.dat')

    #Error counter definition
    no_face_counter = 0

    #Variables that identify the face
    #position in the main frame.
    face_x1 = 0
    face_y1 = 0
    face_x2 = 0
    face_y2 = 0
    face_w = 0
    face_h = 0

    #Variables that identify the ROI
    #position in the main frame.
    roi_x1 = 0
    roi_y1 = 0
    roi_x2 = cam_w
    roi_y2 = cam_h
    roi_w = cam_w
    roi_h = cam_h
    roi_resize_w = int(cam_w/10)
    roi_resize_h = int(cam_h/10)

    while(True):

        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame[roi_y1:roi_y2, roi_x1:roi_x2], cv2.COLOR_BGR2GRAY)

        #Looking for faces with cascade
        #The classifier moves over the ROI
        #starting from a minimum dimension and augmentig
        #slightly based on the scale factor parameter.
        #The scale factor for the frontal face is 1.10 (10%)
        #Scale factor: 1.15=15%,1.25=25% ...ecc
        #Higher scale factors means faster classification
        #but lower accuracy.
        #
        #Return code: 1=Frontal, 2=FrontRotLeft, 
        # 3=FrontRotRight, 4=ProfileLeft, 5=ProfileRight.
        my_cascade.findFace(gray, True, True, True, True, 1.10, 1.10, 1.15, 1.15, 40, 40, rotationAngleCCW=30, rotationAngleCW=-30, lastFaceType=my_cascade.face_type)

        #Accumulate error values in a counter
        if(my_cascade.face_type == 0): 
            no_face_counter += 1

        #If any face is found for a certain
        #number of cycles, then the ROI is reset
        if(no_face_counter == 50):
            no_face_counter = 0
            roi_x1 = 0
            roi_y1 = 0
            roi_x2 = cam_w
            roi_y2 = cam_h
            roi_w = cam_w
            roi_h = cam_h

        #Checking wich kind of face it is returned
        if(my_cascade.face_type > 0):

            #Face found, reset the error counter
            no_face_counter = 0

            #Because the dlib landmark detector wants a precise
            #boundary box of the face, it is necessary to resize
            #the box returned by the OpenCV haar detector.
            #Adjusting the frame for profile left
            if(my_cascade.face_type == 4):
                face_margin_x1 = 20 - 10 #resize_rate + shift_rate
                face_margin_y1 = 20 + 5 #resize_rate + shift_rate
                face_margin_x2 = -20 - 10 #resize_rate + shift_rate
                face_margin_y2 = -20 + 5 #resize_rate + shift_rate
                face_margin_h = -0.7 #resize_factor
                face_margin_w = -0.7 #resize_factor
            #Adjusting the frame for profile right
            elif(my_cascade.face_type == 5):
                face_margin_x1 = 20 + 10
                face_margin_y1 = 20 + 5
                face_margin_x2 = -20 + 10
                face_margin_y2 = -20 + 5
                face_margin_h = -0.7
                face_margin_w = -0.7
            #No adjustments
            else:
                face_margin_x1 = 0
                face_margin_y1 = 0
                face_margin_x2 = 0
                face_margin_y2 = 0
                face_margin_h = 0
                face_margin_w = 0

            #Updating the face position
            face_x1 = my_cascade.face_x + roi_x1 + face_margin_x1
            face_y1 = my_cascade.face_y + roi_y1 + face_margin_y1
            face_x2 = my_cascade.face_x + my_cascade.face_w + roi_x1 + face_margin_x2
            face_y2 = my_cascade.face_y + my_cascade.face_h + roi_y1 + face_margin_y2
            face_w = my_cascade.face_w + int(my_cascade.face_w * face_margin_w)
            face_h = my_cascade.face_h + int(my_cascade.face_h * face_margin_h)

            #Updating the ROI position       
            roi_x1 = face_x1 - roi_resize_w
            if (roi_x1 < 0): roi_x1 = 0
            roi_y1 = face_y1 - roi_resize_h
            if(roi_y1 < 0): roi_y1 = 0
            roi_w = face_w + roi_resize_w + roi_resize_w
            if(roi_w > cam_w): roi_w = cam_w
            roi_h = face_h + roi_resize_h + roi_resize_h
            if(roi_h > cam_h): roi_h = cam_h    
            roi_x2 = face_x2 + roi_resize_w
            if (roi_x2 > cam_w): roi_x2 = cam_w
            roi_y2 = face_y2 + roi_resize_h
            if(roi_y2 > cam_h): roi_y2 = cam_h

            # '''Attributing Cheating to Head_Pose estimation'''
            if (perecieve_left - 10) <= face_x1 <= (perecieve_left + 10) or (perecieve_right - 10) <= face_x1 <= (perecieve_right + 10):
                #If caught looking in left or right Direction
                #print(number_of_seconds_cheating)
                if number_of_seconds_cheating >= 90:
                    f = open(u"HeadPose_log.txt", u"a")
                    print u">>>>>>>>>>>>>>>>>>>>> {}".format(number_of_seconds_cheating)
                    f.write(u">>>>>>>>>>>>>>>>>>>>> {}\n".format(number_of_seconds_cheating))
                    cheating = True
                    now = datetime.datetime.now()
                    f.write(u"Current date and time : {}\n".format(unicode(now.strftime(u"%Y-%m-%d %H:%M:%S"))))
                    f.write(u'Cheating Finally Detected\n\n')
                    f.close()
                    try:
                        if reactive:
                            tts = ALProxy("ALTextToSpeech", ip_addr, port_num)
                            feedback_list = ['please face your work',
                                             'consentrate on your exam',
                                             'please focus on your exam'
                                             'look directly into your own paper']
                            picked = random.sample(feedback_list, 1)[0]
                            tts.say(picked)
                            f = open(u"HeadPose_log.txt", u"a")
                            f.write(picked)
                            f.close()
                        
                    except:
                        f = open(u"HeadPose_log.txt", u"a")
                        f.write(u'\n\n\nthe robot could not be reached!\n')
                        f.close()
                        print('the robot could not be reached!')
                    print u'Cheating Finally Detected'
                    number_of_seconds_cheating = 0
                    # Loop until participant stop cheating
##                time.sleep(1)
                number_of_seconds_cheating += 1
            else:
                potential_cheating = False
                cheating = False
                number_of_seconds_cheating = 0
                        
            #Debugging printing utilities
            if(DEBUG == True):
                print("FACE: ", face_x1, face_y1, face_x2, face_y2, face_w, face_h)
##                print("FACE: ", face_x1, face_x2)
                #print("ROI: ", roi_x1, roi_y1, roi_x2, roi_y2, roi_w, roi_h)
                #Drawing a green rectangle
                # (and text) around the face.
                text_x1 = face_x1
                text_y1 = face_y1 - 3
                if(text_y1 < 0): text_y1 = 0
                cv2.putText(frame, u"FACE", (text_x1,text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1);
                cv2.rectangle(frame, 
                             (face_x1, face_y1), 
                             (face_x2, face_y2), 
                             (0, 255, 0),
                              2)

            #In case of a frontal/rotated face it
            # is called the landamark detector
            if(my_cascade.face_type > 0):
                landmarks_2D = my_detector.returnLandmarks(frame, face_x1, face_y1, face_x2, face_y2, points_to_return=TRACKED_POINTS)

                if(DEBUG == True):
                    #cv2.drawKeypoints(frame, landmarks_2D)

                    for point in landmarks_2D:
                        cv2.circle(frame,( point[0], point[1] ), 2, (0,0,255), -1)


                #Applying the PnP solver to find the 3D pose
                # of the head from the 2D position of the
                # landmarks.
                #retval - bool
                #rvec - Output rotation vector that, together with tvec, brings 
                # points from the model coordinate system to the camera coordinate system.
                #tvec - Output translation vector.
                retval, rvec, tvec = cv2.solvePnP(landmarks_3D, 
                                                  landmarks_2D, 
                                                  camera_matrix, camera_distortion)

                #Now we project the 3D points into the image plane
                #Creating a 3-axis to be used as reference in the image.
                axis = numpy.float32([[50,0,0], 
                                      [0,50,0], 
                                      [0,0,50]])
                imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_distortion)

                #Drawing the three axis on the image frame.
                #The opencv colors are defined as BGR colors such as: 
                # (a, b, c) >> Blue = a, Green = b and Red = c
                #Our axis/color convention is X=R, Y=G, Z=B
                sellion_xy = (landmarks_2D[7][0], landmarks_2D[7][1])
                cv2.line(frame, sellion_xy, tuple(imgpts[1].ravel()), (0,255,0), 3) #GREEN
                cv2.line(frame, sellion_xy, tuple(imgpts[2].ravel()), (255,0,0), 3) #BLUE
                cv2.line(frame, sellion_xy, tuple(imgpts[0].ravel()), (0,0,255), 3) #RED

        # Write the frame into the file 'filename.avi' 
        writer.write(frame)
        
        #Showing the frame and waiting
        # for the exit command
        cv2.imshow(u'Video', frame)
        if cv2.waitKey(1) & 0xFF == ord(u'q'): break
   
    #Release the camera
    cv2.destroyAllWindows()
    video_capture.release()
    writer.release()
    print u"Bye..."





##*************************************************************************##
##**************************** Audio Code *********************************##
##*************************************************************************##
def get_current_time():
    u"""Return Current Time in MS."""

    return int(round(time.time() * 1000))

class MicrophoneStream(object):
    u"""Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self.start_time = get_current_time()
        self.restart_counter = 0
        self.audio_input = []
        self.last_audio_input = []
        self.result_end_time = 0
        self.is_final_end_time = 0
        self.final_request_end_time = 0
        self.bridging_offset = 0
        self.last_transcript_was_final = False
        self.new_stream = True

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        u"""Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield ''.join(data)


def in_my_list(sentence):
    u''' Cheat for cheating phrases in transcribed text'''
    if any (e in sentence for e in cheating_phrases_list):
        return True
    else:
        return False

def get_all_modes(a):
    c = Counter(a)  
    mode_count = max(c.values())
    mode = {key for key, count in c.items() if count == mode_count}
    return mode

def listen_print_loop(responses, stream):
    u"""Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    num_chars_printed = 0
    for response in responses:
        if get_current_time() - stream.start_time > STREAMING_LIMIT:
            stream.start_time = get_current_time()
            break
        
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript
        global speaker_array
        for word in result.alternatives[0].words:
            speaker_array.append(int(word.speaker_tag))
        if speaker_array:
                        speaker_number = mode(get_all_modes(speaker_array))
                        del speaker_array[:]
                        if speaker_array:
                            #print(*speaker_array)
                            pass
                        else:
                            #print("the list is empty")
                            pass

        result_seconds = 0
        result_nanos = 0

        if result.result_end_time.seconds:
            result_seconds = result.result_end_time.seconds

        if result.result_end_time.nanos:
            result_nanos = result.result_end_time.nanos

        stream.result_end_time = int((result_seconds * 1000)
                                     + (result_nanos / 1000000))

        corrected_time = (stream.result_end_time - stream.bridging_offset
                          + (STREAMING_LIMIT * stream.restart_counter))

        
        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = u' ' * (num_chars_printed - len(transcript))

        if not result.is_final:
            stream.is_final_end_time = stream.result_end_time
            stream.last_transcript_was_final = True
           
            num_chars_printed = len(transcript)

        else:
            print u'Speaker {}: {}'.format(speaker_number, transcript + overwrite_chars)
            now = datetime.datetime.now()
            f = open(u"Nao_log.txt", u"a")
            f.write(u"Current date and time : {}\n".format(unicode(now.strftime(u"%Y-%m-%d %H:%M:%S"))))
            f.write(u'Speaker ' + unicode(speaker_number) + u': ' + unicode(transcript) + unicode(overwrite_chars) + u'\n\n')
            f.close()

            # check if cheating phrases are used
            print in_my_list(unicode(transcript))

            if in_my_list(unicode(transcript)):
                print u'Cheating found in text'
                try:
                    if reactive:
                        feedback_list = ['please stop talking',
                                         'do not talk during the exam task',
                                         'talking is not permitted during the exam',
                                         'please dont make noise']
                        picked = random.sample(feedback_list, 1)[0]
                        tts = ALProxy("ALTextToSpeech", ip_addr, port_num)
                        tts.say(picked)
                except:
                    print('the robot could not be reached!')
                
            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(ur'\b(exit|quit)\b', transcript, re.I):
                sys.stdout.write(YELLOW)
                sys.stdout.write(u'Exiting...\n')
                stream.closed = True
                break
                # print('Exiting..')  
            else:
                sys.stdout.write(RED)
                sys.stdout.write(u'\033[K')
                sys.stdout.write(unicode(corrected_time) + u': ' + transcript + u'\r')

                stream.last_transcript_was_final = False

            num_chars_printed = 0


def audio_main():
    f = open(u"Nao_log.txt", u"a")
    f.write(u'##**************************** Audio Log File (Group 1) *********************************##')
    f.close()
    
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = u'en-US'  # a BCP-47 language tag
    
    # If enabled, each word in the first alternative of each result will be
    # tagged with a speaker tag to identify the speaker.
    enable_speaker_diarization = True

    # Optional. Specifies the estimated number of speakers in the conversation.
    #diarization_speaker_count = 2

    client = speech_v1p1beta1.SpeechClient()
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
        enable_speaker_diarization=enable_speaker_diarization)
        
    streaming_config = types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

    with MicrophoneStream(RATE, CHUNK) as stream:
       
        while not stream.closed:
            sys.stdout.write(YELLOW)
            sys.stdout.write(u'\n' + unicode(
                STREAMING_LIMIT * stream.restart_counter) + u': NEW REQUEST\n')

            stream.audio_input = []        
            audio_generator = stream.generator()
                
            requests = (types.StreamingRecognizeRequest(audio_content=content)
                        for content in audio_generator)

            responses = client.streaming_recognize(streaming_config, requests)

                # Now, put the transcription responses to use.
                
            listen_print_loop(responses, stream)

            if stream.result_end_time > 0:
                stream.final_request_end_time = stream.is_final_end_time
            stream.result_end_time = 0
            stream.last_audio_input = []
            stream.last_audio_input = stream.audio_input
            stream.audio_input = []
            stream.restart_counter = stream.restart_counter + 1

            if not stream.last_transcript_was_final:
                sys.stdout.write(u'\n')
            stream.new_stream = True


if __name__ == u"__main__":
    Thread(target = main).start() 
    Thread(target = audio_main).start()
