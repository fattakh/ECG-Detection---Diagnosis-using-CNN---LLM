from transformers import AutoTokenizer, AutoModelForCausalLM

def extract_features(_):
    return {
        "QRS width": "130ms",
        "QT interval": "420ms",
        "PR interval": "180ms",
        "RR interval": "900ms",
        "ST elevation": "2mm",
        "RR pattern": "Irregular"
    }

def build_prompt(features):
    return (
        f"The ECG shows a QRS width of {features['QRS width']}, QT interval of {features['QT interval']}, "
        f"PR interval of {features['PR interval']}, ST elevation of {features['ST elevation']}, "
        f"and {features['RR pattern']} RR intervals. Diagnose the possible cardiac condition, "
        f"provide a follow-up and suggest a medical prescription."
    )

class LLMReasoner:
    def __init__(self):
        model_id = "EleutherAI/gpt-neo-125M"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, use_safetensors=True)

    def query(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs["input_ids"], max_new_tokens=150)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split(prompt)[-1].strip()

def load_llm():
    return LLMReasoner()