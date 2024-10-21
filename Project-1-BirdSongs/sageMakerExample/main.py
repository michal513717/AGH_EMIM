from transformers import pipeline
import librosa

def recognize(filePath):
    
    pipe = pipeline("audio-classification", model="dima806/bird_sounds_classification")

    audio_file = filePath
    audio, sr = librosa.load(audio_file, sr=None)

    result = pipe(audio)

    print(result)


def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    print_hi('PyCharm')
    recognize('path-to-file')