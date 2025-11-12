import PyPDF2 !!
from openai import OpenAI

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text

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

