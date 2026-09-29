import sys
sys.path.insert(0, ".")

from rad_agent import run_radsearch


print("RadSearch AI")
print("Biomedical literature assistant for radiography and medical imaging")
print()


question = input("Ask a research question: ")


print("\nSearching PubMed and reviewing evidence...\n")


result = run_radsearch(question)


final_answer = result["messages"][-1].content


print("=== RADSEARCH AI RESPONSE ===\n")
print(final_answer)