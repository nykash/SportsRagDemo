from ollama import chat
import ollama


class OllamaClient:
    def __init__(self, model: str = "llama3:8b"):
        self.model = model
        self.pulled = False

    def generate(self, messages: list[dict[str, str]]) -> str:
        
        try:
            response = chat(model=self.model, messages=messages)
        except Exception as e:
            # check if stack trace contains "not found"
            if not self.pulled:
                print(f"Model {self.model} not found, pulling...")
                ollama.pull(self.model)
                self.pulled = True
                response = self.generate(messages)
                return response
            else:
                raise e
        return response.message.content

    def generate_single_turn(self, context: str, instruction: str) -> str:
        return self.generate(messages=[
            {
                "role": "system",
                "content": context
            },
            {
                "role": "user",
                "content": instruction
            }
        ])
