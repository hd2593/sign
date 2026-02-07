# Bidirectional Sign Language Translation System
## Components
- Raspberry Pi full conversation demo (fully functional)
  - full.py
  - requirements.txt
  - asl_model(1).h5
- Android Apps (partially functional)
  - landmark_recognition.zip
    - https://drive.google.com/file/d/1C3aiHfJvuNy3U3xztNuLtisCXGti3Lc-/view?usp=sharing
  - chat_history.zip
    - https://drive.google.com/file/d/1C4wPIonSMY-pXbPkJcipPjXkQJFcQHS6/view?usp=sharing
- PC code (working on asl->text acc)
  - pipeline.py

## Start guide
Android Apps: Unzip and open in Android Studio
PC/Raspberry Pi python code:
- install package from requirements.txt
- modify the model path in the code
- export GOOGLE_APPLICATION_CREDENTIALS="*.json"
- export GOOGLE_CLOUD_PROJECT="*"
- python *.py
