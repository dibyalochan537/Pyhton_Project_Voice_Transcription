from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa
model_name = "facebook/wav2vec2-base-960h"
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)
def transcribe_audio(audio_path):
    try:
        speech, sample_rate = librosa.load(audio_path, sr=16000)
        input_values = processor(speech, sampling_rate=sample_rate, return_tensors="pt", padding=True).input_values
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0]) 
        return transcription
    except Exception as e:
        return f"Error during transcription: {str(e)}"
if __name__ == "__main__":
    audio_file = "WhatsApp Audio 2025-01-28 at 13.02.15_5a7c1a39.wav"
    
    print("Transcribing audio...")
    result = transcribe_audio(audio_file)
    print("Transcription:", result)
