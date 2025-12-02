import PyPDF2 # A Python library used to read and extract text from PDF files
from openai import OpenAI # Imports the OpenAI client to interact with the OpenAI API

def extract_text_from_pdf(pdf_path): # Defines a function named extract_text_from_pdf that takes a file path to a PDF as input
    with open(pdf_path, 'rb') as file: # Opens the PDF file in binary read mode ('rb')
        reader = PyPDF2.PdfReader(file) # Creates a PdfReader object to read the contents of the PDF
        text = "" # Initializes an empty string to store the extracted text
        for page in reader.pages: # Loops through each page in the PDF
            page_text = page.extract_text() # Extracts text from the current page
            if page_text: # If text is found on the page...
                text += page_text # ...it appends it to the text string
        return text # Returns the full extracted text from the PDF

pdf_path = "document.pdf" # Specifies the path to the PDF file
try: # Tries to extract text from the PDF using the function defined earlier
    pdf_text = extract_text_from_pdf(pdf_path)
except FileNotFoundError: # If the file is not found, prints an error message in Greek and exits the program    
    print(f"Error: PDF '{pdf_path}' is not found.")
    exit(1)

# Creates a prompt in Greek asking OpenAI to generate 10 multiple-choice questions based on the extracted text
prompt = f"""
Write 10 multiple choice questions about Artificial Intelligence and steepest descent according to the text below. Each question should have 5 possible answers and only one of them should be correct. The audience answering the questions should be at a university level. The questions and answers should be in Greek. The difficulty level of each question is on a Likert scale of 1 to 5 (1 = very easy and 5 = very difficult). Of the 10 questions, one has a difficulty level of 1, two have a difficulty level of 2, three have a difficulty level of 3, two have a difficulty level of 4, and the rest have a difficulty level of 5.

# The prompt includes 3000 characters of the extracted PDF content
Text:
{pdf_text[:3000]}
"""

# Initializes the OpenAI client using your API key
client = OpenAI(
    api_key="enter your API key"
)

# Sends the prompt to the OpenAI API using the gpt-4o-mini model
# The message is sent as a user input in a chat format
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

# Retrieves the generated response (the questions) from the API
output_message = completion.choices[0].message.content

# Opens a file named output.txt in write mode with UTF-8 encoding and writes the generated questions to it
with open("output.txt", "w", encoding="utf-8") as output_file:
    output_file.write(output_message)

# Prints a message confirming that the questions were saved to the file
print("The questions were saved to the file 'output.txt'")


