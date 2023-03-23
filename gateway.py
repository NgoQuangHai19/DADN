from adafruit_api import Adafruit_API
import time
from threading import Thread
from voice import *
def StartVoiceRecognition():
    while True:
        start_signal_detect()

thread = Thread(target = StartVoiceRecognition)
thread.daemon = True

USERNAME = 'haingoquang'
KEY = 'aio_egCZ868jZewBfZs1z4zgPnlB139c'

feed_id_list = ['door','led','fan','led-control']

client = Adafruit_API(USERNAME, KEY, feed_id_list)
client.connect()
thread.start()

counter = 10

while(True):
    
    client.read_serial()
    time.sleep(1)