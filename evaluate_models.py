import os
import glob
import time
from collections import defaultdict

# ----------------------------------------------------------------------
# Setup instructions:
# To use this script, create a folder named 'eval_dataset' in your project root.
# Inside it, create two folders: 'images' and 'audio'.
# Inside each of those, create folders for each emotion you want to test.
# Example:
# eval_dataset/
#   ├── images/
#   │   ├── happy/
#   │   │   ├── face1.jpg
#   │   │   └── face2.png
#   │   └── sad/
#   │       └── face3.jpg
#   └── audio/
#       ├── happy/
#       │   └── voice1.wav
#       └── angry/
#           └── voice2.mp3
# ----------------------------------------------------------------------

# 1. Image Emotion Evaluation (DeepFace)
def evaluate_vision(dataset_path="eval_dataset/images"):
    if not os.path.exists(dataset_path):
        print(f"\n[Vision] Directory '{dataset_path}' not found. Skipping visual evaluation.")
        return

    print("\n" + "="*50)
    print("🎬 Starting Vision Emotion Evaluation (DeepFace)")
    print("="*50)
    
    from deepface import DeepFace
    
    correct = 0
    total = 0
    results_per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    
    # Iterate through emotion folders
    for emotion_dir in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, emotion_dir)
        if not os.path.isdir(folder_path): continue
            
        true_emotion = emotion_dir.lower()
        
        # Iterate through images in the folder
        for file in os.listdir(folder_path):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            
            img_path = os.path.join(folder_path, file)
            total += 1
            results_per_class[true_emotion]["total"] += 1
            
            try:
                # Run DeepFace with mtcnn to bypass opencv cascade issues
                analysis = DeepFace.analyze(img_path, actions=['emotion'], enforce_detection=False, detector_backend='mtcnn', silent=True)
                pred_emotion = analysis[0]['dominant_emotion'].lower() if isinstance(analysis, list) else analysis['dominant_emotion'].lower()
                
                # Check if it matches
                # Note: DeepFace outputs: angry, disgust, fear, happy, sad, surprise, neutral
                if pred_emotion == true_emotion:
                    correct += 1
                    results_per_class[true_emotion]["correct"] += 1
                    print(f"[PASS] {file}: True={true_emotion}, Pred={pred_emotion}")
                else:
                    print(f"[FAIL] {file}: True={true_emotion}, Pred={pred_emotion}")
                    
            except Exception as e:
                print(f"[ERROR] {file}: Failed to process ({str(e)})")

    print("\n--- 📊 Vision Accuracy Report ---")
    if total == 0:
        print("No images found to evaluate.")
        return
        
    print(f"Overall Accuracy: {correct}/{total} ({correct/total*100:.2f}%)")
    for emotion, stats in results_per_class.items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"  - {emotion.capitalize()}: {acc:.2f}% ({stats['correct']}/{stats['total']})")


# 2. Audio Emotion Evaluation (SenseVoice)
def evaluate_audio(dataset_path="eval_dataset/audio"):
    if not os.path.exists(dataset_path):
        print(f"\n[Audio] Directory '{dataset_path}' not found. Skipping audio evaluation.")
        return

    print("\n" + "="*50)
    print("🎙️ Starting Audio Emotion Evaluation (SenseVoice)")
    print("="*50)
    
    from backend.audio_emotion import detect_audio_emotion, _sense_model
    import base64
    
    if not _sense_model:
        print("Error: SenseVoice model failed to load in the backend.")
        return

    correct = 0
    total = 0
    results_per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    
    # Map Sentix emotions to what we expect the folders to be named
    valid_extensions = ('.wav', '.mp3', '.webm', '.ogg')
    
    for emotion_dir in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, emotion_dir)
        if not os.path.isdir(folder_path): continue
            
        true_emotion = emotion_dir.lower()
        
        for file in os.listdir(folder_path):
            if not file.lower().endswith(valid_extensions): continue
            
            file_path = os.path.join(folder_path, file)
            total += 1
            results_per_class[true_emotion]["total"] += 1
            
            try:
                # Convert audio to base64 to simulate frontend payload
                with open(file_path, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode('utf-8')
                
                # Run the backend detector
                pred_emotion = detect_audio_emotion(audio_b64)
                
                if not pred_emotion:
                    pred_emotion = "none detected"
                else:
                    pred_emotion = pred_emotion.lower()
                
                if pred_emotion == true_emotion:
                    correct += 1
                    results_per_class[true_emotion]["correct"] += 1
                    print(f"[PASS] {file}: True={true_emotion}, Pred={pred_emotion}")
                else:
                    print(f"[FAIL] {file}: True={true_emotion}, Pred={pred_emotion}")
                    
            except Exception as e:
                print(f"[ERROR] {file}: Failed to process ({str(e)})")

    print("\n--- 📊 Audio Accuracy Report ---")
    if total == 0:
        print("No audio files found to evaluate.")
        return
        
    print(f"Overall Accuracy: {correct}/{total} ({correct/total*100:.2f}%)")
    for emotion, stats in results_per_class.items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"  - {emotion.capitalize()}: {acc:.2f}% ({stats['correct']}/{stats['total']})")


if __name__ == "__main__":
    print("Welcome to the Sentix Emotion Model Evaluator!")
    print("Checking for 'eval_dataset' directory...")
    
    if not os.path.exists("eval_dataset"):
        os.makedirs("eval_dataset/images/happy", exist_ok=True)
        os.makedirs("eval_dataset/images/sad", exist_ok=True)
        os.makedirs("eval_dataset/audio/joy", exist_ok=True)
        os.makedirs("eval_dataset/audio/sadness", exist_ok=True)
        
        print("\n⚠️ I have created the folder structure for you at: ./eval_dataset/")
        print("Please place your test images and audio files into the respective folders")
        print("named after their TRUE emotion, then run this script again.")
    else:
        evaluate_vision()
        evaluate_audio()
        print("\n✅ Evaluation complete!")
