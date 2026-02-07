# Bidirectional Sign Language Translation System
## Components
- Raspberry Pi full conversation demo (fully functional)
  - full.py
  - requirements.txt
  - asl_model(1).h5
- Android Apps (partially functional)
  - gesture_recognition.apk
  - chat_history.apk
- PC code (working on asl->text acc)
  - pipeline.py

## Start guide
Android Apps: Install seperately
PC/Raspberry Pi python code:
- install package from requirements.txt
- modify the model path in the code
- export GOOGLE_APPLICATION_CREDENTIALS="*.json"
- export GOOGLE_CLOUD_PROJECT="*"
- python *.py
