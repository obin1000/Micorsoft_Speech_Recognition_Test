"""
Speech recognition samples for the Microsoft Cognitive Services Speech SDK
"""
import os
import configparser
config = configparser.ConfigParser()
config.read('../config.ini')

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    print("""Importing the Speech SDK for Python failed. Refer to 
    https://docs.microsoft.com/azure/cognitive-services/speech-service/quickstart-python for installation instructions.
    """)
    import sys
    sys.exit(1)

# Set up the subscription info for the Speech Service:
# Replace with your own subscription key and service region (e.g., "westus").
KEY = config['azure']['SubscriptionKey']
REGION = config['azure']['ServiceRegion']
CURRENTDIR = os.path.dirname(__file__)
AUDIOLOCATION = os.path.join(CURRENTDIR, '..', 'resources', 'audio')

"""
The Speech SDK uses the following format for audio input.
Format  Codec   Bitrate     Sample Rate     Channels
WAV	    PCM	    16-bit      8 or 16 kHz     1 (mono)
"""


class SpeechToText:

    def __init__(self, audio_directory):
        self.speech_config = speechsdk.SpeechConfig(subscription=KEY, region=REGION)
        self.audio_files = self.get_wav_files_from_dir(audio_directory)
        # Example
        print(str(self.speech_recognize_once_from_file(audio_directory + '/Books/Caffaro_gustav.wav')))

    def recognize_all(self):
        pass

    def get_wav_files_from_dir(self, directory):
        """
        Recursively scans the directory for files ending on .wav
        :param directory: path to the directory to scan
        :return: list if all wav files found in given directory and subdirectories
        """
        audio_files = []

        for entry in os.listdir(directory):
            # Create full path for the file/folder
            entry_path = os.path.join(directory, entry)
            # Check if entry is file or folder
            if os.path.isdir(entry_path):
                # if the entry is a subdirectory, call again with the new directory
                print('Sub dir ' + str(entry_path))
                audio_files = audio_files + self.get_wav_files_from_dir(entry_path)
            elif os.path.isfile(entry_path):
                # Check if the file is a wav file
                filename, file_extension = os.path.splitext(entry_path)
                if file_extension == '.wav':
                    audio_files.append(entry_path)
                    print('Found wav file: ' + str(entry))
        return audio_files

    def speech_recognize_once_from_file(self, file):
        """
        Performs one-shot speech recognition with input from an audio file
        :param file: Location of the file to check
        :return:
        """
        audio_config = speechsdk.audio.AudioConfig(filename=file)
        # Creates a speech recognizer using a file as audio input.
        # The default language is "en-us".
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config)

        # Starts speech recognition, and returns after a single utterance is recognized. The end of a
        # single utterance is determined by listening for silence at the end or until a maximum of 15
        # seconds of audio is processed. It returns the recognition text as result.
        # Note: Since recognize_once() returns only a single utterance, it is suitable only for single
        # shot recognition like command or query.
        # For long-running multi-utterance recognition, use start_continuous_recognition() instead.
        result = speech_recognizer.recognize_once()

        # Check the result
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print("Recognized: {}".format(result.text))
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized: {}".format(result.no_match_details))
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print("Speech Recognition canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))
        return result


if __name__ == "__main__":
    converter = SpeechToText(AUDIOLOCATION)
