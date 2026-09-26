import ollama

context = """The next step is to generate a second generation population of solutions from those selected, through a combination of genetic operators: crossover (also called recombination), and mutation.

            For each new solution to be produced, a pair of parent solutions is selected for breeding from the pool selected previously. By producing a child solution using crossover and mutation, a new solution is created which typically shares many characteristics of its parents.

            These processes ultimately result in the next generation population of chromosomes that is different from the initial generation."""

question = input("Enter your question: ")

prompt = f"""Answer using ONLY the context below. If answer not in context, say "I don't know."

Context: {context}

Question: {question}
Answer:"""

answer = ollama.chat(model="qwen3:0.6b",
                    messages=[{"role": "user", "content": prompt}])
print("\nAnswer:", answer["message"]["content"])

cross = input("\nEnter cross-question (or Enter to skip): ")

if cross.strip():
    cross_prompt = f"""Answer using ONLY the context below. If answer not in context, say "I don't know."

Context: {context}
Original Question: {question}
Original Answer: {answer["message"]["content"]}
Cross-Question: {cross}
Answer:"""

    cross_answer = ollama.chat(model="qwen3:0.6b", messages=[{"role": "user", "content": cross_prompt}])
    print("\nCross Answer:", cross_answer["message"]["content"])



# import ollama

# # Find Context
# def find_context():
#     context = """The next step is to generate a second generation population of solutions from those selected, through a combination of genetic operators: crossover (also called recombination), and mutation.

#                  For each new solution to be produced, a pair of parent solutions is selected for breeding from the pool selected previously. By producing a child solution using crossover and mutation, a new solution is created which typically shares many characteristics of its parents.

#                  These processes ultimately result in the next generation population of chromosomes that is different from the initial generation."""
#     return context


# # Task 2: Prompt + Context
# def prompt_context(prompt, context):
#     combined_prompt = f"""Context:{context} Question:{prompt} Answer based on the context above."""
#     return combined_prompt


# # Task 3: Call Ollama
# def use_ollama(prompt):
#     response = ollama.chat(
#         model="qwen3:0.6b",
#         messages=[
#             {"role": "user", "content": prompt}
#         ]
#     )
#     return response["message"]["content"]



# # Main Program
# context = find_context()

# print("CONTEXT ")
# print(context)

# prompt = input("\nEnter your question: ")

# combined_prompt = prompt_context(prompt, context)

# print("\nPROMPT + CONTEXT")
# print(combined_prompt)

# print("\nCALLING OLLAMA ")
# answer = use_ollama(combined_prompt)

# print("\nOUTPUT")
# print(answer)

# # Cross Question
# cross_question = input("\nEnter a cross-question (or press Enter to skip): ")

# if cross_question.strip():
#     cross_prompt = f"""Context:{context} Question:{cross_question} Answer based on the context above."""
#     cross_prompt = f"""Context:{context}

# Original Question: {prompt}
# Original Answer: {answer}

# Cross-Question: {cross_question}

# Answer based on the context above."""

#     cross_answer = use_ollama(cross_prompt)

#     print("\nCROSS-QUESTION OUTPUT")
#     print(cross_answer)