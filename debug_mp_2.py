import mediapipe as mp
try:
    import mediapipe.python.solutions as solutions
    print("Explicit import success")
    print(solutions.pose)
except ImportError as e:
    print(f"ImportError: {e}")
except AttributeError as e:
    print(f"AttributeError: {e}")
