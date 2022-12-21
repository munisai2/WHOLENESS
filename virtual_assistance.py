import pyttsx3
import speech_recognition as sr

ls = sr.Recognizer()
eng = pyttsx3.init()
voices = eng.getProperty('voices')
eng.setProperty('voice', voices[1].id)


# class virtual_assistance(self):

def talk(text):
    eng.say(text)
    eng.runAndWait()


txt = (" Ok !. now, lets start testing in  ")
talk(txt)


def take_command():
    try:
        with sr.Microphone() as source:
            ls.adjust_for_ambient_noise(source)
            print("listening....")
            voice = ls.listen(source)
            command = ls.recognize_google(voice)
            command = 30

    except:
        pass
    return command


def execute():
    command = take_command()
    print(command)
    talk(command)

# execute()


