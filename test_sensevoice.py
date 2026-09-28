from funasr import AutoModel
import sys

def test():
    model_dir = "iic/SenseVoiceSmall"
    print("Loading model...")
    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        remote_code="./model.py",
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device="cpu", # use cpu for testing
    )
    print("Model loaded.")
    # test with the provided example mp3 in the model cache
    # the model path is returned in model.model_path
    res = model.generate(
        input=f"{model.model_path}/example/en.mp3",
        cache={},
        language="auto",  
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,  
        merge_length_s=15,
    )
    print(res)

if __name__ == "__main__":
    test()
