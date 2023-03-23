import speech_recognition as sr
import pyttsx3

r = sr.Recognizer()
def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

def run():
    with sr.Microphone() as source2:
        print('Say something\n<----->')
        r.adjust_for_ambient_noise(source2, duration=0.2)
        print('Ready to record\n<----->')
        SpeakText("I'm here. Please command to me!")
        audio2 = r.listen(source2)
        print('Audio captured\n<----->')
        try:
            MyText = r.recognize_google(audio2, language="vi-VN")
            MyText = MyText.lower()
            print("Did you say ",MyText)

        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
        except sr.UnknownValueError:
            print("Plz try agains")
def start_signal_detect():
    with sr.Microphone() as source2:
        print('Say something\n<----->')
        r.adjust_for_ambient_noise(source2, duration=0.2)
        print('Ready to record\n<----->')
        audio2 = r.listen(source2)
        print('Audio captured\n<----->')
        try:
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()
            if ("hey google" in MyText):
                run()

        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
        except sr.UnknownValueError:
            print("Plz try agains")
start_signal_detect()