import pyttsx3
import datetime
import speech_recognition as sr
from tabulate import tabulate
import GetStock as gs
import sys

engine = pyttsx3.init('sapi5') 
voices= engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def speak(audio):
    engine.say(audio) 
    engine.runAndWait()

def wishme():
    hour = int(datetime.datetime.now().hour)
    if hour >=0 and hour < 12:
        wish = 'Good Morning!'
    elif hour == 12:
        wish = 'Good Noon!'
    elif hour > 12 and hour <= 16:
        wish = 'Good Afternoon!'
    else:
        wish = 'Good Evening!'
    speak(wish + 'How may I help you?')

def takeCommand():
    #It takes microphone input from the user and returns string output

    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 0.8
        audio = r.listen(source)

    try:
        print("Recognizing...")    
        query = r.recognize_google(audio, language='en-in') #Using google for voice recognition.
        speak(f"User said: {query}\n")  #User query will be printed.
        print(f"User said: {query}\n")

    except Exception as e:
        # print(e)    
        query = "Say that again please..."   #Say that again will be printed in case of improper voice
        return None
    return query 
    

# if __name__ == '__main__':
#     wishme()
