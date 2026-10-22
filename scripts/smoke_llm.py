from pathlib import Path
from llama_cpp import Llama

def load_model(model_path: Path, n_ctx: int = 4096 , n_gpu_layers: int = -1)-> Llama:
    if  not model_path.is_file():
        raise FileNotFoundError(f"{model_path} does not exist")

    return Llama(model_path=str(model_path), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=True)

def ask_model(model: Llama, prompt: str):
    response = model.create_chat_completion(
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
    )
    choice = response["choices"][0]
    answer = choice["message"]["content"]
    finish_reason = choice["finish_reason"]

    return answer, finish_reason

def main():
    model_path = Path("./models/Qwen3-8B-Q4_K_M.gguf")
    model = load_model(model_path)

    promt = "how many pigs was in russion story 'Three Little Pigs' ?"
    answer, finish_reason = ask_model(model, promt)
    print(answer)
    print("Finish Reason:", finish_reason)

if __name__ == "__main__":
    main()