"""
Speech recognition samples for the Microsoft Cognitive Services Speech SDK
"""
import datetime
import os
import configparser
import csv
import numpy

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
        self.transcriptions = {}
        # Example
        # print(str(self.speech_recognize_once_from_file(audio_directory + '/Books/Caffaro_gustav.wav')))

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
                # print('Sub dir ' + str(entry_path))
                audio_files = audio_files + self.get_wav_files_from_dir(entry_path)
            elif os.path.isfile(entry_path):
                # Check if the file is a wav file
                filename, file_extension = os.path.splitext(entry_path)
                if file_extension == '.wav':
                    audio_files.append(entry_path)
                    # print('Found wav file: ' + str(entry))
        return audio_files

    def load_transcriptions_into_converter(self, transcription_file, delimiter):
        with open(transcription_file) as csv_file:
            read_csv = csv.reader(csv_file, delimiter=delimiter)
            for row in read_csv:
                if row[0] and row[0] != 'File':
                    self.transcriptions[row[0]] = row[1]
        print(self.transcriptions)

    def calculate_metrics(self):
        for file in self.audio_files:
            transcription = self.retrieve_matching_transcription(file)

            if transcription:
                start = datetime.datetime.now()
                recognized = self.speech_recognize_once_from_file(file).text
                end = datetime.datetime.now()

                real_time_factor = end - start
                print(real_time_factor)

                print("Word error rate", self.wer(transcription.split(), recognized.split()))
                print(recognized, "\n", transcription)

    def calculate_word_error_rate(self, recognized, transcription):
        substitutions = deletions = insertions = corrects = number_of_words = 0
        recognized = recognized.split()
        transcription = transcription.split()

        for i, item in enumerate(transcription):
            if len(recognized) > i:
                if transcription[i].lower() == recognized[i].lower():
                    corrects += 1
        print(corrects)

    def wer(self, r, h):
        """
        Calculation of WER with Levenshtein distance.

        Works only for iterables up to 254 elements (uint8).
        O(nm) time ans space complexity.

        Parameters
        ----------
        r : list
        h : list

        Returns
        -------
        int

        Examples
        --------
        >> wer("who is there".split(), "is there".split())
        1
        >> wer("who is there".split(), "".split())
        3
        >> wer("".split(), "who is there".split())
        3
        """

        d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint8)
        d = d.reshape((len(r) + 1, len(h) + 1))
        for i in range(len(r) + 1):
            for j in range(len(h) + 1):
                if i == 0:
                    d[0][j] = j
                elif j == 0:
                    d[i][0] = i

        # computation
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                if r[i - 1] == h[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitution = d[i - 1][j - 1] + 1
                    insertion = d[i][j - 1] + 1
                    deletion = d[i - 1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)

        return d[len(r)][len(h)]

    def retrieve_matching_transcription(self, audio_file):
        for transcription in self.transcriptions:
            if audio_file.count(transcription):
                return self.transcriptions[transcription]

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
        # if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        #     print("Recognized: {}".format(result.text))
        # elif result.reason == speechsdk.ResultReason.NoMatch:
        #     print("No speech could be recognized: {}".format(result.no_match_details))
        # elif result.reason == speechsdk.ResultReason.Canceled:
        #     cancellation_details = result.cancellation_details
        #     print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        #     if cancellation_details.reason == speechsdk.CancellationReason.Error:
        #         print("Error details: {}".format(cancellation_details.error_details))
        return result


if __name__ == "__main__":
    converter = SpeechToText(AUDIOLOCATION)
    converter.load_transcriptions_into_converter('../resources/text/transcriptions.csv', ';')
    converter.calculate_metrics()
