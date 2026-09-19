import ollama


# ============================================================
# Task 1: Find Context
# ============================================================

def find_context():

    context = """The next step is to generate a second generation population of solutions from those selected, through a combination of genetic operators: crossover (also called recombination), and mutation.

For each new solution to be produced, a pair of parent solutions is selected for breeding from the pool selected previously. By producing a child solution using crossover and mutation, a new solution is created which typically shares many characteristics of its parents.

These processes ultimately result in the next generation population of chromosomes that is different from the initial generation."""

    return context


# ============================================================
# Task 2: Prompt + Context
# ============================================================

def prompt_context(prompt, context):

    combined_prompt = f"""
Context:
{context}

Original Question:
{prompt}
"""

    return combined_prompt


# ============================================================
# Task 3: Use Ollama
# ============================================================

def use_ollama(prompt):

    response = ollama.chat(
        model="qwen3:0.6b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# ============================================================
# Main Program
# ============================================================

context = find_context()

print("========== TASK 1: CONTEXT ==========")
print(context)


prompt = input("\nEnter your prompt: ")

combined_prompt = prompt_context(prompt, context)

print("\n========== TASK 2: PROMPT + CONTEXT ==========")
print(combined_prompt)


print("\n========== TASK 3: CALLING OLLAMA ==========")

answer = use_ollama(combined_prompt)

print("\n========== TASK 4: OUTPUT ==========")
print(answer)


# ============================================================
# Cross Question
# ============================================================

cross_question = input("\nEnter your cross-question: ")

cross_prompt = f"""
Use the following context to answer the user's cross-question.

Context:
{context}

Original Question:
{prompt}

Cross-question:
{cross_question}

Give a simple and clear answer.
"""

cross_answer = use_ollama(cross_prompt)

print("\n========== CROSS-QUESTION OUTPUT ==========")
print(cross_answer)