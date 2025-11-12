from openai import OpenAI # Imports the OpenAI class from the openai library, which allows you to interact with OpenAI's API

# Creates an instance of the OpenAI client using your API key
client = OpenAI(
    api_key="enter your API key"
)

try: # Tries to open the file prompt.txt in read mode with UTF-8 encoding
    with open("prompt.txt", "r", encoding="utf-8") as prompt_file: # Reads the entire content of the file
        prompt = prompt_file.read().strip() # Removes any leading/trailing whitespace using .strip()
except FileNotFoundError: # If the file is not found, prints an error message and exits the program with status code 1
    print("Error: The file 'prompt.txt' was not found. Please check the current working directory and the file name.")
    exit(1)

# Sends the prompt to the OpenAI API using the gpt-4o-mini model
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    store=True, # store=True indicates that the conversation should be stored (if supported by the API)
    messages=[ # The messages parameter simulates a chat conversation, with the user's message containing the prompt
        {"role": "user", "content": prompt}
    ]
)

output_message = completion.choices[0].message.content # Retrieves the generated response from the API — specifically the content of the first choice

with open("output.txt", "w", encoding="utf-8") as output_file: # Opens a file named output.txt in write mode with UTF-8 encoding
    output_file.write(output_message) # Writes the generated response to this file

print("Output has been written to output.txt") # Prints a message confirming that the output was successfully saved


