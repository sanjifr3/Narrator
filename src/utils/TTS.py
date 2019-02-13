# -*- coding: utf-8 -*-
"""
Wrapper class for the Amazon Polly TTS
"""
from boto3 import Session
from botocore.exceptions import BotoCoreError, ClientError
from contextlib import closing
import pygame


class TTS(object):
    """Module for generate text to speech files."""
    # TTS params
    region = 'us-east-1'
    speech_rate = "40%"  # 'medium' #'slow'
    sentence_break = 1000
    paragraph_break = 1200
    voice = "Joanna"  # "Ivy"
    output_format = "ogg_vorbis"

    def __init__(self, sounds_dir='.'):
        """Construct the TTS class."""
        # Create AWS polly session
        session = Session(profile_name='default')
        self.polly = session.client('polly')
        self.sounds_dir = sounds_dir

    def play_audio(self, path, wait=True):
        """Playback audio files."""
        # Initialize pygame if not already initialized
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        channel = pygame.mixer.Channel(5)
        sound = pygame.mixer.Sound(path)
        channel.play(sound)
        if wait:
            while channel.get_busy():
                pass

    def generate_audio(self, text, sound_file=None):
        """Generates audio from a given text file."""
        try:
            resp = self.polly.synthesize_speech(
                Text=text, OutputFormat=self.output_format,
                VoiceId=self.voice)
        except (BotoCoreError, ClientError) as error:
            print(error)
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
        """
        Convert text to speech.

        Args:
            text: Line to say
            sound_file: Path to save aduio file too
            wait: Hold code or return
        Return:
            success of generating file
        """
        if not sound_file:
            sound_file = self.sounds_dir + 'sound.ogg'
        success = self.generate_audio(text, sound_file)
        if success:
            self.play_audio(sound_file, wait)
        return success
