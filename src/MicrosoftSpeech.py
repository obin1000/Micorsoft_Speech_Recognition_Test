"""
Speech recognition samples for the Microsoft Cognitive Services Speech SDK
"""
import datetime
import os
import configparser
import csv
import time

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
        """
        loads transcriptions from file into memory
        :param transcription_file: path to the transcription file
        :param delimiter: delimiter to delimit csv columns
        """

        with open(transcription_file) as csv_file:
            read_csv = csv.reader(csv_file, delimiter=delimiter)
            for row in read_csv:
                if row[0] and row[0] != 'File':
                    self.transcriptions[row[0]] = row[1]

    def calculate_metrics(self):
        metrics = {}
        for file in self.audio_files:
            transcription = self.retrieve_matching_transcription(file)
            metrics[file] = {}

            if transcription:

                start = time.time()
                recognized = self.speech_recognize_once_from_file(file).text
                end = time.time()

                passed_seconds = float(end - start)
                metrics[file]['real_time_factor'] = round((passed_seconds / len(file)), 3)
                metrics[file]['word_error_rate'] = self.wer(transcription[0], recognized)[0]
                metrics[file]['word_correct_rate'] = self.wer(transcription[0], recognized)[1]
                metrics[file]['transcription_used'] = transcription[1]
                print("Recognized: " + recognized + "\nTranscription: " + transcription[0])
                print(metrics[file])
            else:
                metrics[file] = 'No transcription was found for this audio file.'

    def wer(self, ref, hyp, debug=False):
        r = ref.split()
        h = hyp.split()
        # costs will holds the costs, like in the Levenshtein distance algorithm
        costs = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]
        # backtrace will hold the operations we've done.
        # so we could later backtrace, like the WER algorithm requires us to.
        backtrace = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]

        OP_OK = 0
        OP_SUB = 1
        OP_INS = 2
        OP_DEL = 3

        DEL_PENALTY = 1  # Tact
        INS_PENALTY = 1  # Tact
        SUB_PENALTY = 1  # Tact
        # First column represents the case where we achieve zero
        # hypothesis words by deleting all reference words.
        for i in range(1, len(r) + 1):
            costs[i][0] = DEL_PENALTY * i
            backtrace[i][0] = OP_DEL

        # First row represents the case where we achieve the hypothesis
        # by inserting all hypothesis words into a zero-length reference.
        for j in range(1, len(h) + 1):
            costs[0][j] = INS_PENALTY * j
            backtrace[0][j] = OP_INS

        # computation
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                if r[i - 1] == h[j - 1]:
                    costs[i][j] = costs[i - 1][j - 1]
                    backtrace[i][j] = OP_OK
                else:
                    substitutionCost = costs[i - 1][j - 1] + SUB_PENALTY  # penalty is always 1
                    insertionCost = costs[i][j - 1] + INS_PENALTY  # penalty is always 1
                    deletionCost = costs[i - 1][j] + DEL_PENALTY  # penalty is always 1

                    costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                    if costs[i][j] == substitutionCost:
                        backtrace[i][j] = OP_SUB
                    elif costs[i][j] == insertionCost:
                        backtrace[i][j] = OP_INS
                    else:
                        backtrace[i][j] = OP_DEL

        # back trace though the best route:
        i = len(r)
        j = len(h)
        numSub = 0
        numDel = 0
        numIns = 0
        numCor = 0
        if debug:
            print("OP\tREF\tHYP")
            lines = []
        while i > 0 or j > 0:
            if backtrace[i][j] == OP_OK:
                numCor += 1
                i -= 1
                j -= 1
                if debug:
                    lines.append("OK\t" + r[i] + "\t" + h[j])
            elif backtrace[i][j] == OP_SUB:
                numSub += 1
                i -= 1
                j -= 1
                if debug:
                    lines.append("SUB\t" + r[i] + "\t" + h[j])
            elif backtrace[i][j] == OP_INS:
                numIns += 1
                j -= 1
                if debug:
                    lines.append("INS\t" + "****" + "\t" + h[j])
            elif backtrace[i][j] == OP_DEL:
                numDel += 1
                i -= 1
                if debug:
                    lines.append("DEL\t" + r[i] + "\t" + "****")
        if debug:
            lines = reversed(lines)
            for line in lines:
                print(line)
            print("Ncor " + str(numCor))
            print("Nsub " + str(numSub))
            print("Ndel " + str(numDel))
            print("Nins " + str(numIns))
            return (numSub + numDel + numIns) / (float)(len(r))

        wer = round((numSub + numDel + numIns) / (float)(len(r)), 3)
        wcr = round((numCor / len(r)), 3)
        return wer, wcr

    def retrieve_matching_transcription(self, audio_file):
        for transcription in self.transcriptions:
            if audio_file.count(transcription):
                return self.transcriptions[transcription], transcription

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
