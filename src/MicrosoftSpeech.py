"""
Speech recognition samples for the Microsoft Cognitive Services Speech SDK
"""

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    print("""
    Importing the Speech SDK for Python failed. Refer to 
    https://docs.microsoft.com/azure/cognitive-services/speech-service/quickstart-python for installation instructions.
    """)
    import sys

    sys.exit(1)

# Set up the subscription info for the Speech Service:
# Replace with your own subscription key and service region (e.g., "westus").
speech_key, service_region = "YourSubscriptionKey", "YourServiceRegion"

AUDIOLOCATION = 'resources/audio/'


class SpeechToText:
    def __init__(self, audio_directory):
        self.audio_directory = audio_directory
        self.speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

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


if __name__ == "__main__":
    converter = SpeechToText(AUDIOLOCATION)
