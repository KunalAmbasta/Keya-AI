#!/usr/bin/python

from os import environ, path
from cv2 import *
import RPi.GPIO as GPIO
import pyaudio
import os,sys
import time
import aiml
import pyowm
from espeak import espeak
import wikipedia
import duckduckgo
import speech_recognition as sr
from speech_recognition import Microphone
from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *
import glob, random, sys, vlc, time, argparse
import datetime
from datetime import datetime
from datetime import datetime as date
from time import localtime, strftime
import cv2, numpy
from dht11 import dht11


folder = '/home/pi/Music/mp3'
files = glob.glob(folder+"/*.mp3")
if len(files) == 0:
    print("No mp3 files found", folder, "..exciting")
    sys.exit(1)
random.shuffle(files)

player = vlc.MediaPlayer()
medialist = vlc.MediaList(files)
mlplayer = vlc.MediaListPlayer()
mlplayer.set_media_player(player)
mlplayer.set_media_list(medialist)



def handle_changed_track(event, player):
    media = player.get_media()
    media.parse()
    artist = media.get_meta(vlc.Meta.Artist) or "Unknown artist"
    title = media.get_meta(vlc.Meta.Title) or "Unknown song title"
    album = media.get_meta(vlc.Meta.Album) or "Unknown album"
    print(title+"\n"+artist+" - "+album)

def recognize_people():
    
    size = 4
    #fn_haar = '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml'
    fn_haar = 'keyaData/haars/haarcascade_frontalface_default.xml'
    fn_dir = 'keyaData/att_faces'
    # Part 1: Create fisherRecognizer
    print('Training...')
    # Create a list of images and a list of corresponding names
    (images, lables, names, id) = ([], [], {}, 0)
    for (subdirs, dirs, files) in os.walk(fn_dir):
        for subdir in dirs:
            names[id] = subdir
            subjectpath = os.path.join(fn_dir, subdir)
            for filename in os.listdir(subjectpath):
                path = subjectpath + '/' + filename
                lable = id
                images.append(cv2.imread(path, 0))
                lables.append(int(lable))
            id += 1
    (im_width, im_height) = (112, 92)

    # Create a Numpy array from the two lists above
    (images, lables) = [numpy.array(lis) for lis in [images, lables]]

    # OpenCV trains a model from the images
    # NOTE FOR OpenCV2: remove '.face'
    model = cv2.createFisherFaceRecognizer()
    model.train(images, lables)

    # Part 2: Use fisherRecognizer on camera stream
    haar_cascade = cv2.CascadeClassifier(fn_haar)
    webcam = cv2.VideoCapture(0)
    for j in range (1,11):
        (rval, frame) = webcam.read()
        frame=cv2.flip(frame,1,0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mini = cv2.resize(gray, (gray.shape[1] / size, gray.shape[0] / size))
        faces = haar_cascade.detectMultiScale(mini)
        for i in range(len(faces)):
            face_i = faces[i]
            (x, y, w, h) = [v * size for v in face_i]
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (im_width, im_height))

            # Try to recognize the face
            prediction = model.predict(face_resize)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            username=names[prediction[0]]
            print(username)
            # Write the name of recognized face
            # [1]
            cv2.putText(frame,
                '%s - %.0f' % (names[prediction[0]],prediction[1]),
                (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
        cv2.imshow('OpenCV', frame)
    
        key = cv2.waitKey(10) & 0xff
        if key == 27:
            break
    
    # Close the window
    webcam.release()
 
    # De-allocate any associated memory usage
    cv2.destroyAllWindows() 

    for i in range (1,5):
        cv2.waitKey(1)
    try:
        cm = 'espeak -v f3 -k5 "hello dear "'+username+'"! How are you?"'
        answer(cm)
    except:
        cm = 'espeak -v f3 -k5 "Sorry I cannot recognize. Either no one is there or face is not in my memory."'
        answer(cm)

    p.terminate()
    streaming()


def take_pictures():
    cm = 'espeak -v f3 -k5 "Get Ready for the photoshoot. Please look at the camera."'
    answer(cm)
    camera = cv2.VideoCapture(0)
    ts = int(time.time())
    return_value,image = camera.read()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('image',gray)
    time.sleep(2)
    cv2.imwrite('/home/pi/Pictures/'+str(ts)+'.jpg',image)
    time.sleep(2)
    camera.release()
    cm = 'espeak -v f3 -k5 "Picture successfully captured. It is stored in Pictures folder."'
    answer(cm)
    cv2.destroyAllWindows()
    for i in range (1,5):
        cv2.waitKey(1)

#MODELDIR = "/usr/share/pocketsphinx/model/hmm"

# Init decoder
config = Decoder.default_config()
config.set_string('-hmm', 'keyaData/en-us/en-us')
config.set_string('-dict', 'keyaData/keya_rpi3.dic')
config.set_float('-kws_threshold', 1e-20)
decoder = Decoder(config)

# Add searches
decoder.set_keyphrase('keyphrase', 'HEY KEYA')
decoder.set_jsgf_file('jsgf', 'keyaData/keya_rpi3.jsgf')
decoder.set_search('keyphrase')

# Create the kernel and learn AIML files
kernel = aiml.Kernel()
kernel.learn("keyaData/std-startup.xml")
kernel.respond("load aiml b")


global p, stream
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
stream.start_stream()

def songStream():
    try:
        stream.close()
        p.terminate()
        print("Stream is closed and pyaudio has been terminated. Press button 1 to start streaming ")
        cm = 'espeak -v f3 -k5 "stream successfully closed. press button 1 to restart."'
        answer(cm)
    except:
        pass

def ans(cm):
    try:
        os.system(cm)
    except:
        pass

def answer(cm):
    try:
        stream.stop_stream()
        os.system(cm)
        stream.start_stream()
    except:
        pass

def streaming():
    try:
        global p, stream
        print("streaming block reached")
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True,  frames_per_buffer=4096)
        stream.start_stream()
        print("streaming block crossed")
    except:
        pass

def keya_date():
    try:
        dt = list(time.localtime())
        print(dt)
        #cm = 'espeak -v f3 -k5 "Today is year "'+str(dt[0])+'" month "'+str(dt[1])+'" day "'+str(dt[2])
        cm = 'espeak -v f3 -k5 "Today is "'+time.strftime("%A")+time.strftime("%d")+time.strftime("%B")+time.strftime("%Y")
        answer(cm)
    except:
        pass

def keya_day():
    try:
        day = date.today().strftime("%A")
        print day
        cm = 'espeak -v f3 -k5 "Today is "'+day
        answer(cm)
    except:
        pass

def keya_time():
    try:
        dt = list(time.localtime())
        #dt = datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
        print(dt)
        cm = 'espeak -v f3 -k5 " Time is "'+time.strftime("%I")+'" "'+time.strftime("%M")+'" "'+time.strftime("%p")
        answer(cm)
    except:
        pass

def keya_aiml(word):
    try:
        keya_response = kernel.respond(word)
        cm = 'flite -voice slt -t "'+keya_response+' "'
        answer(cm)
    except:
        pass

def article():
    stream.close()
    try:
        r = sr.Recognizer()
        with sr.Microphone(device_index=None, sample_rate=None, chunk_size=1024) as source:
            r.adjust_for_ambient_noise(source)
            print("Set minimum energy threshold to {}".format(r.energy_threshold))
            cm = 'flite -voice slt -t " Sir, tell me what to search on wikipedia? Results are powered by Wikipedia"'
            ans(cm)
            print("Say something!")
            audio = r.listen(source)
        WIT_AI_KEY = "SKRYF6M56EKT6TNCDFPSHWHC3X46ALG6" # Wit.ai keys are 32-character uppercase alphanumeric strings
        try:
            #witreply = r.recognize_wit(audio, key=WIT_AI_KEY)
            witreply = r.recognize_google(audio)
            print("Wit.ai thinks you said " + witreply)
        except sr.UnknownValueError:
            print("Wit.ai could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Wit.ai service; {0}".format(e))

        witreplied = str(witreply)
        cm = 'flite -voice slt -t " I am searching for"'+ witreplied + '" .This may take a while. Please Wait!"'
        ans(cm)
        if witreplied and not witreplied.isspace():
            try:
                para = wikipedia.summary(witreply, sentences=2)
                paragraph = para.replace('"', '\\"')
                #ny = wikipedia.page(witreply)
                print(str(paragraph))
                cm = 'flite -voice slt -t " '+str(paragraph)+'"'
                ans(cm)
            except:
                cm = 'flite -voice slt -t " Sorry! I did not understand what you said."'
                ans(cm)
        else:
            cm = 'flite -voice slt -t " Sorry! I did not understand what you said."'
            ans(cm)
    except:
        print("nothing found")
        cm = 'flite -voice slt -t " Sorry dear user! Right now I am unable to connect to internet."'
        ans(cm)

    p.terminate()
    streaming()

def internet():
    stream.close()
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            print("Set minimum energy threshold to {}".format(r.energy_threshold))
            cm = 'flite -voice slt -t " Sir, tell me what to search on internet? Search results are powered by DuckDuckGo"'
            ans(cm)
            print("Say something!")
            audio = r.listen(source)


        # recognize speech using Wit.ai
        WIT_AI_KEY = "SKRYF6M56EKT6TNCDFPSHWHC3X46ALG6" # Wit.ai keys are 32-character uppercase alphanumeric strings
        try:
            #witreply = r.recognize_wit(audio, key=WIT_AI_KEY)
            witreply = r.recognize_google(audio)
            print("Wit.ai thinks you said " + witreply)
        except sr.UnknownValueError:
            print("Wit.ai could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Wit.ai service; {0}".format(e))

        witreplied = str(witreply)
        cm = 'flite -voice slt -t " I am searching for '+ witreplied + ' .This may take a while. Please Wait!"'
        ans(cm)
        if witreplied and not witreplied.isspace():
            try:
                reply = duckduckgo.search(""+witreply)
                if(reply.type) == 'article':
                    art = reply.abstract.text
                    article = art.replace('"', '\\"')
                    cm = 'flite -voice slt -t " '+str(article)+'"'
                    ans(cm)
            except:
                print("nothing found")
                cm = 'flite -voice slt -t " Sorry! I did not understand what you said."'
                ans(cm)

        else:
            cm = 'flite -voice slt -t " Sorry! I did not understand what you said."'
            ans(cm)
    except:
        print("nothing found")
        cm = 'flite -voice slt -t " Sorry dear user! Right now I am unable to connect to internet."'
        ans(cm)
        

    p.terminate()
    streaming()


def weather():
    try:
        owm = pyowm.OWM("6a7fffa90f922c31d37890c63f243fe8")
        observation = owm.weather_at_id(1266414)
        w = observation.get_weather()
        cl = str(w.get_clouds())
        rain = str(w.get_rain())
        press = str(w.get_pressure()['press'])
        hum = str(w.get_humidity())
        temp = str(w.get_temperature(unit='celsius')['temp'])
        det = str(w.get_detailed_status())
        cm = 'flite -voice slt -t "Current Weather report, cloud coverage is '+cl+' percent, atomspheric pressure '+press+' and humidity '+hum+' percent and overall status of weather is '+det+' " '
        ans(cm)
    except:
        cm = 'flite -voice slt -t " Sorry! Dear user. I am having problem while connecting to server to fetch weather information."'
        ans(cm)
    
    

def dhtSensor():
    try:
        time.sleep(1)
        print "Inside dht try block"
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        # read data using pin 05 BCM format
        instance = dht11.DHT11(pin=5)
        time.sleep(1)
        print "About to read data"
        for i in range(0,3):
            result = instance.read()
            time.sleep(1)
            print result
            if result.is_valid():
                break
        if result.is_valid():
            print "entered into loop"
            print("Temperature: %d C" % result.temperature)
            print("Humidity: %d %%" % result.humidity)
            temp = str(result.temperature)
            hum = str(result.humidity)
            cm = 'espeak -v f3 -k5   "Right Now humidity is '+hum+'  percent and temperature is '+temp+'  degree celsius  "'
            #cm = 'espeak -v f3 -k5 "Temperature is "'+str(temperature)+'" degree celcius and Humidity is "'+str(humidity)+'" percent"'
            answer(cm)
        else :
            print result.error_code
            cm = 'espeak -v f3 -k5 "Right Now I am unable to check your local weather condition "'
            #cm = 'espeak -v f3 -k5 "Temperature is "'+str(temperature)+'" degree celcius and Humidity is "'+str(humidity)+'" percent"'
            answer(cm)
        time.sleep(1)
    except:
        pass



FAN_PIN = 16
LED_PIN = 20
COOLER_PIN = 21
COFFEE_PIN = 6
TV_PIN = 19
FILTER_PIN = 26
WASHING_PIN = 13
RED_LED = 22
GREEN_LED = 27
YELLOW_LED = 17

def GPIOsetup():
    try:
        GPIO.setwarnings(False) 
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(FAN_PIN, GPIO.OUT)
        GPIO.setup(LED_PIN, GPIO.OUT)
        GPIO.setup(COOLER_PIN, GPIO.OUT)
        GPIO.setup(COFFEE_PIN, GPIO.OUT)
        GPIO.setup(TV_PIN, GPIO.OUT)
        GPIO.setup(FILTER_PIN, GPIO.OUT)
        GPIO.setup(FILTER_PIN, GPIO.OUT)
        GPIO.setup(RED_LED, GPIO.OUT)
        GPIO.setup(GREEN_LED, GPIO.OUT)
        GPIO.setup(YELLOW_LED, GPIO.OUT)
    except:
        pass
    
def redOFF():
    try:
        #configuration for normally closed connection
        GPIOsetup()
        GPIO.output(RED_LED, GPIO.LOW) #red off
    except:
        pass


def redON():
    try:
        #configuration for normally closed connection
        GPIOsetup()
        GPIO.output(RED_LED, GPIO.HIGH) #red on
    except:
        pass

def greenOFF():
    try:
        #configuration for normally closed connection
        GPIOsetup()
        GPIO.output(GREEN_LED, GPIO.LOW) #green off
    except:
        pass


def greenON():
    try:
        #configuration for normally closed connection
        GPIOsetup()
        GPIO.output(GREEN_LED, GPIO.HIGH) #green on
    except:
        pass

def yellowOFF():
    try:
        #configuration for normally closed connection
        GPIOsetup()
        GPIO.output(YELLOW_LED, GPIO.LOW) #yellow off
    except:
        pass


def yellowON():
    try:
        #configuration for normally closed connection
        GPIOsetup()
        GPIO.output(YELLOW_LED, GPIO.HIGH) #yellow on
    except:
        pass


def fanOFF():
    try:
        #configuration for normally closed connection
        GPIOsetup()
        GPIO.output(FAN_PIN, GPIO.HIGH) #fan on
    except:
        pass


def fanON():
    try:
        #configuration for normally closed connection
        GPIOsetup()
        GPIO.output(FAN_PIN, GPIO.LOW) #fan off
    except:
        pass


def ledOFF():
    try:
        #configuration for normally closed connection
        GPIOsetup()
        GPIO.output(LED_PIN, GPIO.HIGH) #fan on
    except:
        pass


def ledON():
    try:
        #configuration for normally closed connection
        GPIOsetup()
        GPIO.output(LED_PIN, GPIO.LOW) #fan off
    except:
        pass


def coolerOFF():
    try:
        #configuration for normally closed connection
        GPIOsetup()
        GPIO.output(COOLER_PIN, GPIO.HIGH) #fan on
    except:
        pass


def coolerON():
    try:
        #configuration for normally closed connection
        GPIOsetup()
        GPIO.output(COOLER_PIN, GPIO.LOW) #fan off
    except:
        pass

def coffeeOFF():
    try:
        #configuration for normally closed connection
        GPIOsetup()
        GPIO.output(COFFEE_PIN, GPIO.HIGH) #fan on
    except:
        pass


def coffeeON():
    try:
        #configuration for normally closed connection
        GPIOsetup()
        GPIO.output(COFFEE_PIN, GPIO.LOW) #fan off
    except:
        pass

def tvOFF():
    try:
        #configuration for normally closed connection
        GPIOsetup()
        GPIO.output(TV_PIN, GPIO.HIGH) #fan on
    except:
        pass


def tvON():
    try:
        #configuration for normally closed connection
        GPIOsetup()
        GPIO.output(TV_PIN, GPIO.LOW) #fan off
    except:
        pass

def filterOFF():
    try:
        #configuration for normally closed connection
        GPIOsetup()
        GPIO.output(FILTER_PIN, GPIO.HIGH) #fan on
    except:
        pass


def filterON():
    try:
        #configuration for normally closed connection
        GPIOsetup()
        GPIO.output(FILTER_PIN, GPIO.LOW) #fan off
    except:
        pass

def washingOFF():
    try:
        #configuration for normally closed connection
        GPIOsetup()
        GPIO.output(WASHING_PIN, GPIO.HIGH) #fan on
    except:
        pass


def washingON():
    try:
        #configuration for normally closed connection
        GPIOsetup()
        GPIO.output(WASHING_PIN, GPIO.LOW) #fan off
    except:
        pass




def headerSetup():
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(23, GPIO.IN,pull_up_down=GPIO.PUD_UP)
        GPIO.setup(24, GPIO.IN,pull_up_down=GPIO.PUD_UP)
        GPIO.setup(25, GPIO.IN,pull_up_down=GPIO.PUD_UP)
    except:
        GPIO.cleanup()
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(23, GPIO.IN,pull_up_down=GPIO.PUD_UP)
        GPIO.setup(24, GPIO.IN,pull_up_down=GPIO.PUD_UP)
        GPIO.setup(25, GPIO.IN,pull_up_down=GPIO.PUD_UP)

def shutdown():
    print("SHUT DOWN NOW")
    greenOFF()
    redON()
    cm = 'espeak -v f3 -k5 "Shutting down the system in next five seconds"'
    answer(cm)
    time.sleep(5)
    os.system("sudo shutdown -h now")
    redOFF()
    greenON()

def reboot():
    print("Reboot NOW")
    greenOFF()
    redON()
    cm = 'espeak -v f3 -k5 "Rebooting system in next five seconds"'
    answer(cm)
    time.sleep(5)
    os.system("sudo reboot")
    redOFF()
    greenON()

def poweroff():
    try:
        #configuration for normally closed connection
        GPIOsetup()
        fanOFF()
        ledOFF()
        coolerOFF()
        coffeeOFF()
        tvOFF()
        filterOFF()
        washingOFF()
    except:
        pass


    
def main():
    poweroff()
    redOFF()
    yellowOFF()
    greenON()
    print("KEYA is ready to help you")
    time.sleep(2)
    cm = 'espeak -v f3 -k5 " Hello dear user, Welcome! I am KEYA, your personal assistant. Please tell me, how can I help you?" '
    answer(cm)
    in_speech_bf = False
    decoder.start_utt()
    while True:
        try:
            headerSetup()
            inputValue1 = GPIO.input(23)
            if (inputValue1 == False):
                print("Button 1 pressed. Shutting down the system in next five seconds ")
                shutdown()


            inputValue2 = GPIO.input(24)
            if (inputValue2 == False):
                print("Button 2 pressed. Restarting Keya ")
                main()

            inputValue3 = GPIO.input(25)
            if (inputValue3 == False):
                redOFF()
                greenON()
                mlplayer.pause()
                #vmlplayer.pause()
                streaming()
                print("Button 3 pressed. Pyaudio and stream has been started ")
        except:
            pass
                
            
        try:
            buf = stream.read(4096, exception_on_overflow=False)
        except:
            pass
        if buf:
            decoder.process_raw(buf, False, False)
            if decoder.get_in_speech() != in_speech_bf:
                in_speech_bf = decoder.get_in_speech()
                if not in_speech_bf:
                    decoder.end_utt()

                    try:
                        # Print hypothesis and switch search to another mode
                        print('Result:', decoder.hyp().hypstr)
                        word = str(decoder.hyp().hypstr)
                        playerem = player.event_manager()
                        playerem.event_attach(vlc.EventType.MediaPlayerMediaChanged,handle_changed_track, player)
                        if(word == "WAKE UP"):
                            greenOFF()
                            redON()
                            cm = 'espeak -v f3 -k5 "Always for you sir, Please tell me How may I assist you?"'
                            answer(cm)
                            redOFF()
                            greenON()

                        elif(word == "WHAT IS THE DATE TODAY" or word == "KEYA WHAT IS THE DATE NOW"):
                            greenOFF()
                            redON()
                            keya_date()
                            redOFF()
                            greenON()

                        elif(word == "WHAT IS THE DAY TODAY" or word == "KEYA WHAT IS THE DAY NOW"):
                            greenOFF()
                            redON()
                            keya_day()
                            redOFF()
                            greenON()

                        elif(word == "WHAT IS THE TIME NOW"):
                            greenOFF()
                            redON()
                            keya_time()
                            redOFF()
                            greenON()

                        elif(word == "CHECK INFORMATION ON INTERNET"):
                            greenOFF()
                            yellowON()
                            stream.close()
                            internet()
                            greenON()
                            yellowOFF()

                        elif(word == "IS INTERNET CONNECTED"):
                            greenOFF()
                            yellowON()
                            conin()
                            greenON()
                            yellowOFF()

                        elif(word == "CHECK INFORMATION ON WIKIPEDIA"):
                            greenOFF()
                            print("loop crossed")
                            yellowON()
                            print("loop crossed")
                            stream.close()
                            article()
                            greenON()
                            yellowOFF()

                        elif(word == "PLAY SONG" or word == "PLAY MP THREE SONG"):
                            print("Pressed play button")
                            greenOFF()
                            redON()
                            songStream()
                            cm = 'espeak -v f3 -k5 "Playing songs for you, Sir! from music folder."'
                            answer(cm)
                            mlplayer.play()
                            redON()

                        elif(word == "PAUSE SONG" or word == "PAUSE MP THREE SONG"):
                            print("Pressed pause button")
                            greenOFF()
                            redON()
                            cm = 'espeak -v f3 -k5 "Paused song. Anything else for you, Sir! "'
                            answer(cm)
                            mlplayer.pause()
                            redOFF()
                            greenON()
                            
                        elif(word == "STOP SONG" or word == "STOP MP THREE SONG"):
                            print("Pressed stop button")
                            greenOFF()
                            redON()
                            cm = 'espeak -v f3 -k5 "Stopped songs. What else can I do for you, Sir!"'
                            answer(cm)
                            mlplayer.stop()
                            random.shuffle(files)
                            medialist = vlc.MediaList(files)
                            mlplayer.set_media_list(medialist)
                            redOFF()
                            greenon()
                
                        elif(word == "LAST SONG" or word == "LAST MP THREE SONG"):
                            greenOFF()
                            redON()
                            songStream()
                            cm = 'espeak -v f3 -k5 "Playing last song for you, Sir! from music folder."'
                            answer(cm)
                            print("Pressed back button")
                            mlplayer.previous()
                            redON()

                        elif(word == "NEXT SONG" or word == "NEXT MP THREE SONG"):
                            greenOFF()
                            redON()
                            songStream()
                            cm = 'espeak -v f3 -k5 "Playing next song for you, Sir! from music folder."'
                            answer(cm)
                            print("Pressed forward button")
                            mlplayer.next()
                            redON()

                        elif(word == "PLEASE INCREASE VOLUME"):
                            greenOFF()
                            redON()
                            cm = 'espeak -v f3 -k5 "Ok Sir, I have increased volume and set it to hundred percent"'
                            answer(cm)
                            print("increasing Volume")
                            os.system('amixer  sset PCM,0 100%')
                            redOFF()
                            greenON()

                        elif(word == "PLEASE DECREASE VOLUME"):
                            greenOFF()
                            redON()
                            cm = 'espeak -v f3 -k5 "Ok Sir, I have decreased volume and set it to ninty percent"'
                            answer(cm)
                            print("decreasing Volume")
                            os.system('amixer  sset PCM,0 90%')
                            redOFF()
                            greenON()

                        elif(word == "KEYA PLEASE SET VOLUME FOR SONG"):
                            greenOFF()
                            redON()
                            cm = 'espeak -v f3 -k5 "Ok Sir, I have set volume to ninty five percent."'
                            answer(cm)
                            print("setting Volume for song")
                            os.system('amixer  sset PCM,0 95%')
                            redOFF()
                            greenON()


                        elif(word == "PLEASE CAPTURE PHOTO" or word == "PLEASE SHOOT PHOTO"):
                            greenOFF()
                            redON()
                            print("CAPTURE PHOTO")
                            take_pictures()
                            redOFF()
                            greenON()

                        elif(word == "WHATS THE WEATHER FORECASTING"):
                            greenOFF()
                            yellowON()
                            print("Checking weather forcasting")
                            cm = 'espeak -v f3 -k5 " dear user! Please wait. This may take a while" '
                            answer(cm)
                            weather()
                            greenON()
                            yellowOFF()

                        elif(word == "PLEASE CHECK LOCAL WEATHER CONDITION" or word == "PLEASE CHECK TEMPERATURE CONDITION"):
                            print("Checking weather condition")
                            greenOFF()
                            redON()
                            cm = 'espeak -v f3 -k5 " dear user! Please wait. This may take a while" '
                            answer(cm)
                            dhtSensor()
                            redOFF()
                            greenON()
      
                        elif(word == "PLEASE POWER ON ELECTRIC FAN"):
                            print("FAN TURNED ON")
                            greenOFF()
                            redON()
                            fanON()
                            cm = 'espeak -v f3 -k5 " Ok sir, Fan has switched on. Anything else for you, SIR? " '
                            answer(cm)
                            redOFF()
                            greenON()

                        elif(word == "PLEASE POWER OFF ELECTRIC FAN" or word == "PLEASE STOP ELECTRIC FAN"):
                            print("FAN TURNED OFF")
                            greenOFF()
                            redON()
                            fanOFF()
                            cm = 'espeak -v f3 -k5 " Ok sir, Fan has switched off. Anything else for you, SIR? " '
                            answer(cm)
                            redOFF()
                            greenON()
                        
                        elif(word == "PLEASE POWER ON LIGHT" or word == "PLEASE POWER ON LED LIGHT"):
                            print("LED LIGHT TURNED ON")
                            greenOFF()
                            redON()
                            ledON()
                            cm = 'espeak -v f3 -k5 " Ok sir, L E D LIGHT has switched on. Anything else for you, SIR? " '
                            answer(cm)
                            redOFF()
                            greenON()

                        elif(word == "PLEASE POWER OFF LIGHT"  or word == "PLEASE STOP LIGHT"):
                            print("LED LIGHT TURNED OFF")
                            greenOFF()
                            redON()
                            ledOFF()
                            cm = 'espeak -v f3 -k5 " Ok sir, L E D light has switched off. Anything else for you, SIR? " '
                            answer(cm)
                            redOFF()
                            greenON()
                        
                        elif(word == "PLEASE POWER ON AIR COOLER" or word == "PLEASE POWER ON ELECTRIC COOLER"):
                            print("AIR COOLER TURNED ON")
                            greenOFF()
                            redON()
                            coolerON()
                            cm = 'espeak -v f3 -k5 " Ok sir, Air Cooler has switched on. Anything else for you, SIR? " '
                            answer(cm)
                            redOFF()
                            greenON()

                        elif(word == "PLEASE STOP AIR COOLER" or word == "PLEASE POWER OFF ELECTRIC COOLER"  or word == "PLEASE STOP ELECTRIC COOLER"):
                            print("AIR COOLER TURNED OFF")
                            greenOFF()
                            redON()
                            coolerOFF()
                            cm = 'espeak -v f3 -k5 " Ok sir, Air Cooler has switched off. Anything else for you, SIR? " '
                            answer(cm)
                            redOFF()
                            greenON()

                        elif(word == "PLEASE POWER ON COFFEE MAKER MACHINE"):
                            print("COFFEE MAKER MACHINE TURNED ON")
                            greenOFF()
                            redON()
                            coffeeON()
                            cm = 'espeak -v f3 -k5 " Ok sir. Coffee maker machine has switched on. Anything else for you, SIR? " '
                            answer(cm)
                            redOFF()
                            greenON()

                        elif(word == "PLEASE POWER OFF COFFEE MAKER MACHINE"  or word == "PLEASE STOP COFFEE MAKER MACHINE"):
                            print("COFFEE MAKER MACHINE TURNED OFF")
                            greenOFF()
                            redON()
                            coffeeOFF()
                            cm = 'espeak -v f3 -k5 " Ok sir, Coffee maker machine has turned off. Anything else for you, SIR? " '
                            answer(cm)
                            redOFF()
                            greenON()

                        elif(word == "PLEASE POWER ON WASHING MACHINE"):
                            print("WASHING MACHINE TURNED ON")
                            greenOFF()
                            redON()
                            washingON()
                            cm = 'espeak -v f3 -k5 " Ok sir, Washing Machine has switched on. Anything else for you, SIR? " '
                            answer(cm)
                            redOFF()
                            greenON()

                        elif(word == "PLEASE POWER OFF WASHING MACHINE"  or word == "PLEASE STOP WASHING MACHINE"):
                            print("WASHING MACHINE TURNED OFF")
                            greenOFF()
                            redON()
                            washingOFF()
                            cm = 'espeak -v f3 -k5 " Ok sir, Washing machine has turned off. Anything else for you, SIR? " '
                            answer(cm)
                            redOFF()
                            greenON()

                        elif(word == "PLEASE POWER ON TV" or word == "PLEASE POWER ON TELEVISION"):
                            print("TV TURNED ON")
                            greenOFF()
                            redON()
                            tvON()
                            cm = 'espeak -v f3 -k5 " Ok sir, TV has switched on. Anything else for you, SIR? " '
                            answer(cm)
                            redOFF()
                            greenON()

                        elif(word == "PLEASE POWER OFF TV" or word == "PLEASE POWER OFF TELEVISION" or word == "PLEASE STOP TV"):
                            print("TV TURNED OFF")
                            greenOFF()
                            redON()
                            tvOFF()
                            cm = 'espeak -v f3 -k5 " Ok sir, TV has switched off. Anything else for you, SIR? " '
                            answer(cm)
                            redOFF()
                            greenON()

                        elif(word == "PLEASE POWER ON WATER FILTER"):
                            print("WATER FILTER TURNED ON")
                            greenOFF()
                            redON()
                            filterON()
                            cm = 'espeak -v f3 -k5 " Ok sir, Water Filter has switched on. Anything else for you, SIR? " '
                            answer(cm)
                            redOFF()
                            greenON()

                        elif(word == "PLEASE POWER OFF WATER FILTER" or word == "PLEASE STOP WATER FILTER"):
                            print("WATER FILTER TURNED OFF")
                            greenOFF()
                            redON()
                            filterOFF()
                            cm = 'espeak -v f3 -k5 " Ok sir, water filter has switched off. Anything else for you, SIR? " '
                            answer(cm)
                            redOFF()
                            greenON()


                        elif(word == "PLEASE POWER OFF EVERYTHING" or word == "PLEASE STOP EVERYTHING"):
                            print("EVERYTHING TURNED OFF")
                            greenOFF()
                            redON()
                            poweroff()
                            cm = 'espeak -v f3 -k5 " Ok sir, all the connected devices have switched off. Anything else for you, SIR? " '
                            answer(cm)
                            redOFF()
                            greenON()

                        elif(word == "KEYA PLEASE SHUTDOWN NOW"):
                            print("SHUT DOWN NOW")
                            greenOFF()
                            redON()
                            cm = 'espeak -v f3 -k5 "Shutting down the system in next five seconds"'
                            answer(cm)
                            time.sleep(5)
                            os.system("sudo shutdown -h now")
                            redOFF()
                            greenON()

                        elif(word == "KEYA PLEASE REBOOT NOW"):
                            print("REBOOT NOW")
                            greenOFF()
                            redON()
                            cm = 'espeak -v f3 -k5 "Rebooting the system in next five seconds"'
                            answer(cm)
                            time.sleep(5)
                            os.system("sudo reboot")
                            redOFF()
                            greenON()

                        elif(word == "KEYA PLEASE SUSPEND"):
                            print("SUSPEND NOW")
                            greenOFF()
                            redON()
                            cm = 'espeak -v f3 -k5 "Closing the running app or program"'
                            answer(cm)
                            time.sleep(1)
                            sys.exit()

                        elif(word == "KEYA PLEASE SLEEP"):
                            print("SLEEPING NOW")
                            greenOFF()
                            redON()
                            cm = 'espeak -v f3 -k5 "Going to sleep now. You can wake me up by pressing button 1."'
                            answer(cm)
                            time.sleep(1)
                            songStream()

                        elif(word == "IDENTIFY PERSON USING YOUR CAMERA"):
                            print("Initializing webcam")
                            greenOFF()
                            redON()
                            cm = 'espeak -v f3 -k5 "Initializing camera to recognize person"'
                            answer(cm)
                            stream.close()
                            recognize_people()
                            redOFF()
                            greenON()
                    

                               
                        else:
                            greenOFF()
                            redON()
                            keya_aiml(word)
                            redOFF()
                            greenON()
                            

                            

                    except:
                        pass
                    
                    if decoder.get_search() == 'keyphrase' and decoder.hyp() != None and decoder.hyp().hypstr == 'HEY KEYA':
                        cm = 'espeak -v f3 -k5 "Yes Sir, I am listening."'
                        answer(cm)
                        decoder.set_search('jsgf')
                    else:
                        decoder.set_search('keyphrase')

                    decoder.start_utt()
        else:
            break
    decoder.end_utt()



if __name__ == '__main__':
    main()

