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

pdf_path = "document.pdf"
try:
    pdf_text = extract_text_from_pdf(pdf_path)
except FileNotFoundError:
    print(f"Error: Το αρχείο PDF '{pdf_path}' δεν βρέθηκε.")
    exit(1)

prompt = f"""
Γράψε 10 ερωτήσεις πολλαπλής επιλογής για την Τεχνητή Νοημοσύνη και steepest descent σύμφωνα με το κείμενο που ακολουθεί. Κάθε ερώτηση να έχει 5 πιθανές απαντήσεις και μόνο μια από αυτές να είναι σωστή. Το κοινό που απαντά στις ερωτήσεις είναι πανεπιστημιακού επιπέδου. Οι ερωτήσεις και οι απαντήσεις να είναι στα ελληνικά. Το επίπεδο δυσκολίας κάθε ερώτησης είναι σε κλίμακα Likert 1 ως 5 (1 = πολύ εύκολη και 5 = πολύ δύσκολη). Από τις 10 ερωτήσεις, η μια να έχει επίπεδο δυσκολίας 1, οι δυο να έχουν επίπεδο δυσκολίας 2, οι τρεις να έχουν επίπεδο δυσκολίας 3, οι δυο να έχουν επίπεδο δυσκολίας 4, και οι υπόλοιπες να έχουν επίπεδο δυσκολίας 5.


Κείμενο:
{pdf_text[:3000]}
"""

client = OpenAI(
    api_key="enter your API key"
)

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

output_message = completion.choices[0].message.content

with open("output.txt", "w", encoding="utf-8") as output_file:
    output_file.write(output_message)

print("Οι ερωτήσεις αποθηκεύτηκαν στο αρχείο 'output.txt'")



