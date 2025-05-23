from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./model")
model = AutoModelForCausalLM.from_pretrained("./model")

print("Start chatting with your Gainesville Restaurant Bot (type 'exit' to quit)")

while True:
    try:
        prompt = input("You: ")
        if prompt.lower() == "exit":
            print("Goodbye!")
            break
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(
            **inputs,
            max_length=150,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        # Remove echo of prompt from response
        answer = response.replace(prompt, "").strip()
        print("Bot:", answer)
    except Exception as e:
        print("⚠️ Error:", str(e))