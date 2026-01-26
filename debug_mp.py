import mediapipe as mp
try:
    print(mp.solutions)
    print("Success")
except AttributeError as e:
    print(e)
    print(mp.__file__)
