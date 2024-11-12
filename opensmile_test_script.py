"""
Save this as test_opensmile.py and run it from the command line with:
python test_opensmile.py
"""
import os
import sys
import opensmile
import soundfile as sf
import numpy as np

def verify_opensmile_binary():
    """Verify OpenSMILE binary is accessible"""
    try:
        # Create a simple test WAV file
        sample_rate = 16000
        duration = 1  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Save test audio
        test_file = "test_audio.wav"
        sf.write(test_file, audio_data, sample_rate)
        
        print("Step 1: Test audio file created")
        
        # Try to create Smile object with minimal configuration
        smile = opensmile.Smile(
            feature_set='GeMAPSv01b',
            feature_level='functionals',
            sampling_rate=sample_rate,
        )
        
        print("Step 2: Smile object created")
        
        # Try to process the test file
        features = smile.process_file(test_file)
        
        print("Step 3: Features extracted successfully")
        print(f"Number of features: {len(features.columns)}")
        
        # Clean up
        os.remove(test_file)
        
        return True, features
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print(f"Error type: {type(e)}")
        
        # Additional debugging information
        print("\nEnvironment information:")
        print(f"Python version: {sys.version}")
        print(f"OpenSMILE version: {opensmile.__version__}")
        print(f"PATH: {os.environ.get('PATH')}")
        
        return False, None

def check_audio_file(file_path):
    """Check if audio file can be read"""
    try:
        data, samplerate = sf.read(file_path)
        print(f"\nAudio file information:")
        print(f"Sample rate: {samplerate}")
        print(f"Duration: {len(data)/samplerate:.2f} seconds")
        print(f"Shape: {data.shape}")
        return True
    except Exception as e:
        print(f"\nError reading audio file: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting OpenSMILE test...")
    
    # First test with synthetic audio
    success, features = verify_opensmile_binary()
    
    if success:
        print("\nBasic OpenSMILE functionality verified!")
        
        # Now test with actual audio file if provided
        if len(sys.argv) > 1:
            audio_file = sys.argv[1]
            print(f"\nTesting with provided audio file: {audio_file}")
            
            if check_audio_file(audio_file):
                try:
                    smile = opensmile.Smile(
                        feature_set='GeMAPSv01b',
                        feature_level='functionals',
                    )
                    features = smile.process_file(audio_file)
                    print("\nSuccessfully processed provided audio file!")
                    print(f"Number of features: {len(features.columns)}")
                except Exception as e:
                    print(f"\nError processing provided audio file: {str(e)}")
    else:
        print("\nOpenSMILE test failed. Please check the error messages above.")