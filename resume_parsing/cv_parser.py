import utils.pdf2text as pdf2text
import spacy
from spacy.matcher import Matcher
import re
import pandas as pd
import multiprocessing as mp
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
from pathlib import Path

# Load pre-trained model
nlp = spacy.load('en_core_web_sm')

# Initialize matcher with a vocab
matcher = Matcher(nlp.vocab)


def extract_name(resume_text):
    nlp_text = nlp(resume_text)

    # First name and Last name are always Proper Nouns
    pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]

    matcher.add('NAME', patterns=[pattern])

    matches = matcher(nlp_text)

    for match_id, start, end in matches:
        span = nlp_text[start:end]
        if 'name' not in span.text.lower():
            matcher.remove('NAME')
            # Extract the name and format it
            name_parts = [token.text.capitalize() for token in span]
            return ' '.join(name_parts)
    matcher.remove('NAME')
    return None



def extract_mobile_number(text):
    mob_num_regex = r'''(0)?(\+91)?[-\s]?(\d{3}[-.\s]??\d{3}[-.\s]??\d{4}|\(\d{3}\) [-.\s]*\d{3}[-.\s]??\d{4}|\d{3}[-.\s]??\d{4})'''
    phone = re.findall(re.compile(mob_num_regex), text)

    if phone:
        number = ''.join(phone[0])
        if len(number) > 10:
            return '+' + number
        else:
            return number
    return None


def extract_email(email):
    email = re.findall(r"([^@|\s]+@[^@]+\.[^@|\s]+)", email)

    if email:
        try:
            return email[0].split()[0].strip(';')
        except IndexError:
            return None
    return None


def check_skills(word, skills_data):
    for skill in skills_data:
        if str(word).lower() == str(skill).lower():
            return str(skill)
    return False


def extract_skills(resume_text):
    nlp_text = nlp(resume_text)

    # Removing stop words and implementing word tokenization
    tokens = [token.text for token in nlp_text if not token.is_stop]

    data = pd.read_csv('./utils/skills_db.txt', header=None, encoding='utf-8')

    # Extract values
    skills_data = data[0].tolist()

    pool = mp.Pool(mp.cpu_count())

    skills = [pool.apply_async(check_skills, args=(
        str(word), skills_data)) for word in nlp_text.noun_chunks]

    token_skills = [pool.apply_async(check_skills, args=(
        str(word), skills_data)) for word in tokens]

    skills.extend(token_skills)

    skills = [p.get() for p in skills if p.get() is not False]

    pool.close()
    pool.join()

    return list(set(skills))


def process_resume(resume_path):
    # Extract information from the resume
    cv_text = pdf2text.get_Text(resume_path)
    name = extract_name(cv_text)
    mobile_number = extract_mobile_number(cv_text)
    email = extract_email(cv_text)
    skills = extract_skills(cv_text)

    # Generate PDF filename
    filename = Path(resume_path).stem + "_information.pdf"

    # Save information to PDF
    save_to_pdf(name, mobile_number, email, skills, filename)

# Assuming the resumes are stored in the 'resumes' directory
resumes_directory = "resumes"

def save_to_pdf(name, mobile_number, email, skills, filename, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create the full path for the output PDF file
    output_path = os.path.join(output_dir, filename)

    # Create PDF file and write resume information
    c = canvas.Canvas(output_path, pagesize=letter)
    c.setFontSize(11)
    c.drawString(100, 750, name if name else "Not found")
    c.drawString(100, 730, "Mobile Number: " + (mobile_number if mobile_number else "Not found"))
    c.drawString(100, 710, "Email: " + (email if email else "Not found"))

    # Display skills
    if skills:
        max_commas_per_line = 5  # Maximum commas per line
        skills_lines = []
        line = "Skills: "
        comma_count = 0
        for skill in skills:
            if comma_count >= max_commas_per_line:
                skills_lines.append(line.rstrip(", "))
                line = "   "  # Add some space before continuing the skills list
                comma_count = 0
            line += skill + ", "
            comma_count += 1
        skills_lines.append(line.rstrip(", "))

        y_coordinate = 690
        for line in skills_lines:
            c.drawString(100, y_coordinate, line)
            y_coordinate -= 20

    c.save()


if __name__ == '__main__':
    # Output directory
    output_dir = "output"

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each resume in the directory
    for filename in os.listdir(resumes_directory):
        if filename.endswith(".pdf"):
            resume_path = os.path.join(resumes_directory, filename)
            cv_text = pdf2text.get_Text(resume_path)
            name = extract_name(cv_text)
            mobile_number = extract_mobile_number(cv_text)
            email = extract_email(cv_text)
            skills = extract_skills(cv_text)

            # Generate PDF filename
            output_filename = os.path.splitext(filename)[0] + "_information.pdf"

            # Save information to PDF in the output directory
            save_to_pdf(name, mobile_number, email, skills, output_filename, output_dir)