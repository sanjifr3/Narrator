from boto3 import Session
from botocore.exceptions import BotoCoreError, ClientError
from contextlib import closing
import os
import pygame
import sys
import subprocess
from tempfile import gettempdir
import hashlib
import pandas as pd

class TTS(object):
  # TTS params
  region = 'us-east-1'
  speech_rate = "40%" # 'medium' #'slow'
  sentence_break = 1000
  paragraph_break = 1200
  voice = "Joanna" # "Ivy"
  output_format = "ogg_vorbis"
  
  def __init__(self, sounds_dir='.'):
    # Create AWS polly session
    session = Session(profile_name='default')
    self.polly = session.client('polly')
    self.sounds_dir = sounds_dir

  def playAudio(self, path, wait=True):
    # Initialize pygame if not already initialized
    if not pygame.mixer.get_init():
      pygame.mixer.init()

    channel = pygame.mixer.Channel(5)
    sound = pygame.mixer.Sound(path)
    channel.play(sound)
    if wait:
      while channel.get_busy():
        pass

  def generateAudio(self, text, sound_file=None):
    try:
      resp = self.polly.synthesize_speech(Text=text,
                                    OutputFormat=self.output_format,
                                    VoiceId=self.voice)
    except (BotoCoreError, ClientError) as error:
      print (error)
      return False

    if not sound_file:
      sound_file = self.sounds_dir + 'sound.ogg'

    # Access the audio stream in the response
    if "AudioStream" in resp:
      with closing(resp['AudioStream']) as stream:
        try:
          with open(sound_file, "wb") as file:
            file.write(stream.read())
        except IOError as error:
          print(error)
          return False

    return True

  def speak(self, text, sound_file=None, wait=True):
    if not sound_file:
      sound_file = self.sounds_dir + 'sound.ogg'
    success = self.generateAudio(text, sound_file)
    if success:
      self.playAudio(sound_file, wait)
    return success
