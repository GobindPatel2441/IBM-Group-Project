import os
import sys
from pathlib import Path
import pandas as pd
import difflib

# Add backend to path to import translator properly
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from translator import translate_to_english, translate_from_english

def main():
    test_csv = BASE_DIR.parent / "Translator Model" / "translator" / "datasets" / "processed" / "test.csv"
    if not test_csv.exists():
        print("Test CSV not found.")
        return

    df = pd.read_csv(test_csv)
    
    # Lang mapping from code to name
    lang_map = {
        "hin_Deva": "Hindi",
        "ory_Orya": "Odia",
        "mar_Deva": "Marathi",
        "tel_Telu": "Telugu",
        "tam_Taml": "Tamil",
        "mal_Mlym": "Malayalam",
        "kan_Knda": "Kannada",
        "guj_Gujr": "Gujarati",
        "pan_Guru": "Punjabi",
        "ben_Beng": "Bengali",
    }

    results = []
    
    print("Testing Translation Integration...\n")
    
    for code, name in lang_map.items():
        # Get 2 random medium length sentences
        subset = df[(df['lang'] == code) & (df['category'] == 'medium')].dropna()
        if len(subset) >= 2:
            samples = subset.sample(2, random_state=42)
        else:
            continue
            
        print(f"--- {name} ---")
        avg_score = 0
        for _, row in samples.iterrows():
            source = str(row['source_text'])
            target = str(row['target_text'])
            
            # Note: in test.csv, target_text is usually English.
            # Let's cleanly translate source to English
            try:
                translated = translate_to_english(source, name)
                
                similarity = difflib.SequenceMatcher(None, target.lower(), translated.lower()).ratio() * 100
                avg_score += similarity
                
                print(f"Original Text: {source.encode('utf-8', 'ignore').decode('ascii', 'ignore')}")
                print(f"Expected Eng : {target}")
                print(f"Actual Eng   : {translated}")
                print(f"Similarity   : {similarity:.1f}%")
                print("-")
            except Exception as e:
                print(f"Error: {e}")
        
        print(f"Average Similarity for {name}: {avg_score/2:.1f}%\n")

if __name__ == "__main__":
    main()
