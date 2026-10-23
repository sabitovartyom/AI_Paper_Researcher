from pathlib import Path

from llama_cpp import Llama
from llama_cpp.llama_types import ChatCompletionRequestMessage


DEFAULT_SYSTEM_MESSAGE = (
    "Отвечай на русском языке только по предоставленному фрагменту. "
    "Если ответа в нём нет, скажи: "
    "'В предоставленном фрагменте этой информации нет'."
)


def load_model(
    model_path: Path,
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
) -> Llama:
    if not model_path.is_file():
        raise FileNotFoundError(f"{model_path} does not exist")

    return Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=True,
    )


def build_messages(
    user_question: str,
    source_text: str,
    system_message: str = DEFAULT_SYSTEM_MESSAGE,
) -> list[ChatCompletionRequestMessage]:
    if not user_question.strip():
        raise ValueError("Вопрос не должен быть пустым.")
    if not source_text.strip():
        raise ValueError("Фрагмент источника не должен быть пустым.")

    return [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": (
                f"Фрагмент:\n{source_text}\n\n"
                f"Вопрос: {user_question}\n"
                "/no_think"
            ),
        },
    ]


def ask_model(
    model: Llama,
    user_question: str,
    source_text: str,
    system_message: str = DEFAULT_SYSTEM_MESSAGE,
) -> tuple[str, str | None]:
    response = model.create_chat_completion(
        messages=build_messages(
            user_question=user_question,
            source_text=source_text,
            system_message=system_message,
        ),
        max_tokens=1024,
    )
    choice = response["choices"][0]
    answer = choice["message"]["content"]
    finish_reason = choice["finish_reason"]

    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Модель не вернула текстовый ответ.")

    return answer, finish_reason


def main() -> None:
    model_path = Path("./models/Qwen3-8B-Q4_K_M.gguf")
    model = load_model(model_path)

    source_text = (
        "В эксперименте использовали модель с 8 миллиардами параметров. "
        "Вычисления выполняли на одной видеокарте с 24 ГиБ памяти. "
        "Название видеокарты и скорость генерации не указаны."
    )
    questions = [
        "Сколько параметров у модели?",
        "Какую скорость генерации получили?",
    ]

    for user_question in questions:
        answer, finish_reason = ask_model(model, user_question, source_text)
        print("Вопрос:", user_question)
        print("Ответ:", answer)
        print("Finish Reason:", finish_reason)
        print()


if __name__ == "__main__":
    main()
