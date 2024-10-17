import json
import re
import random
import signal
from src.medrag import MedRAG


# Load the benchmark JSON file
with open('medqa.json', 'r') as f:
    benchmark_data = json.load(f)

# Get all questions
# all_questions = list(benchmark_data.items())

# Limit to the first 200 questions
# all_questions = all_questions[:1000]
# all_questions = all_questions[600:1000]
# all_questions = all_questions[400:600]
# all_questions = all_questions[600:800]
# all_questions = all_questions[800:1000]

# Get random questions
all_questions = random.sample(list(benchmark_data.items()), 1)

# Initialize the MedRAG system
rag = MedRAG(llm_name="axiong/PMC_LLaMA_13B", rag=True, retriever_name="MedCPT", corpus_name="Textbooks", HNSW=True)

# Store the results of comparisons
results = []
correct_count = 0
answered_questions = 0
number_all_questions = 0

# Define a timeout handler
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Timeout occurred while generating answer.")

# Set the timeout limit to 60 seconds
signal.signal(signal.SIGALRM, timeout_handler)

# Iterate over each question and get the generated answer
for question_id, question_data in all_questions:
    # Extract the question, options, and correct answer
    question = question_data['question']
    options = question_data['options']
    correct_answer = question_data['answer']

    number_all_questions += 1
    # Use MedRAG to generate the answer with a timeout
    # signal.alarm(120)  # Set alarm for 60 seconds
    try:
        # Use MedRAG to generate the answer
        answer = rag.medrag_answer(question=question, options=options, k=1)
        
        # Debugging: Check the type and raw content of the answer
        print(f"Answer Type: {type(answer)}")
        print(f"Generated Answer (Raw): {answer}")
        
        if isinstance(answer, tuple):
            predicted_answer = answer[0]  # The first element contains the predicted answer
            documents = answer[1]  # Retrieved documents
            logits = answer[2]  # Logits for the answer

            # Log the retrieved documents for debugging purposes (optional)
            print(f"Retrieved Documents: {documents}")
            print(f"Logits: {logits}")

            # Extract the option from the generated answer using a regular expression
            match = re.search(r"OPTION\s([A-Z])", predicted_answer)
            
            if match:
                generated_choice = match.group(1)  # Extracted option letter, e.g., 'A', 'B', 'C', etc.
            else:
                print(f"Invalid predicted answer: {predicted_answer}. Logging and continuing.")
                generated_choice = None
        else:
            generated_choice = None

        if not generated_choice:
            print(f"No valid answer choice extracted for question ID: {question_id}")
            continue
        
        print(f"generated_choice: {generated_choice}")
        # Compare the generated answer with the correct one
        is_correct = correct_answer == generated_choice
        if is_correct:
            correct_count += 1
        
        answered_questions += 1
        
        accuracy = correct_count / answered_questions * 100 if answered_questions > 0 else 0
        print(f"Correct Answer: {correct_answer}")
        print(f"Is Correct: {is_correct}")
        print(f"Current Accuracy: {accuracy:.2f}%")        
        print(f"All Questions(Answered Questions): {number_all_questions}({answered_questions})")

    except TimeoutException:
        print(f"Skipping question ID: {question_id} due to timeout.")
        continue


