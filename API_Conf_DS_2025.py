from openai import OpenAI # Imports the OpenAI class from the openai library, which allows you to interact with OpenAI's API

# Creates an instance of the OpenAI client using my API key
client = OpenAI(
    api_key="enter your API key"
)

try:
    with open("prompt.txt", "r", encoding="utf-8") as prompt_file:
        prompt = prompt_file.read().strip()
except FileNotFoundError:
    print("Error: The file 'prompt.txt' was not found. Please check the current working directory and the file name.")
    exit(1)

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    store=True,
    messages=[
        {"role": "user", "content": prompt}
    ]
)

output_message = completion.choices[0].message.content

with open("output.txt", "w", encoding="utf-8") as output_file:
    output_file.write(output_message)

print("Output has been written to output.txt")

