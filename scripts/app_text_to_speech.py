import importlib
import logging as logger

import pyttsx3

from scripts.app_environment import os_running_environment, tts_speed, translate_docs, translate_dst

if os_running_environment == 'mac':
    driver = "pyttsx3.drivers.nsss"
    importlib.import_module(driver)
elif os_running_environment == 'windows':
    driver = 'pyttsx3.drivers.sapi5'
    importlib.import_module(driver)
elif os_running_environment == 'linux':
    driver = 'pyttsx3.drivers.espeak'
    importlib.import_module(driver)

engine = pyttsx3.init()
engine.setProperty('rate', tts_speed)
voices = engine.getProperty('voices')  # getting details of current voice


def print_all_voices_helper():
    logger.log(logger.DEBUG, "On this system following voices are supported:")
    for index, voice in enumerate(voices):
        logger.log(logger.DEBUG, f"Voice {index}: ID: {voice.id}, Name: {voice.name}, Languages: {voice.languages}, Gender: {voice.gender}, Age: {voice.age}")
        engine.setProperty('voice', voice.id)


print_all_voices_helper()


def stop_voice():
    engine.stop()


def supported_voices():
    """
    first retrieves a sub-dictionary based on translate_dst,
    defaulting to English if the given language is not found.
    It then retrieves a voice id based on os_running_environment
    from the sub-dictionary, defaulting to English if the
    given OS environment is not found.
    :return: voice from dictionary
    """
    voice_dict = {
        'hr': {
            'mac': voices[74].id,
            'windows': voices[2].id,
            'linux': voices[0].id,  # Adjust voice index as per your requirement for Linux
        },
        'en': {
            'mac': voices[0].id,
            'windows': voices[0].id,
            'linux': voices[0].id,
        },
        # Add more languages as needed
    }

    # If the translate_dst is not found, default to 'en'
    os_voice_dict = voice_dict.get(translate_dst, voice_dict['en'])

    # Use `os_running_environment` as key to get the corresponding voice id
    # If the key is not found in the dictionary, defaults to English (voices[0].id)
    return os_voice_dict.get(os_running_environment, voices[0].id)


def speak_chunk(content):
    if not translate_docs:
        # defaults to English
        voice = voices[0].id
    else:
        voice = supported_voices()

    engine.setProperty('voice', voice)
    engine.say(content)
    engine.runAndWait()
