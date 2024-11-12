"""
Save this as test_audio.py and run it with:
python test_audio.py negative.wav
"""
import soundfile as sf
import os

def test_audio_file(file_path):
    """Test if audio file can be read and print its properties"""
    print(f"Testing audio file: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    
    try:
        data, samplerate = sf.read(file_path)
        print("\nAudio file successfully read!")
        print(f"Sample rate: {samplerate} Hz")
        print(f"Duration: {len(data)/samplerate:.2f} seconds")
        print(f"Number of channels: {1 if len(data.shape) == 1 else data.shape[1]}")
        print(f"Data type: {data.dtype}")
        return True
    except Exception as e:
        print(f"\nError reading audio file: {str(e)}")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_audio_file(sys.argv[1])
    else:
        print("Please provide an audio file path as argument")