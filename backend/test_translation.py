from translator import translate_to_english, translate_from_english

def main():
    print("Testing translation setup:")
    
    # Simple Hindi word
    text = "नमस्ते दुनिया"
    eng = translate_to_english(text, "Hindi")
    print(f"Hindi -> English: {text.encode('utf-8', 'ignore')} -> {eng}")
    
    # Simple English phrase
    orig = "How are you doing today?"
    hin = translate_from_english(orig, "Hindi")
    print(f"English -> Hindi: {orig} -> {hin.encode('utf-8', 'ignore')}")

if __name__ == "__main__":
    main()
