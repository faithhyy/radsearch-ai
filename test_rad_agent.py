import sys
sys.path.insert(0, ".")

from rad_agent import run_radsearch


question = """
What does research say about occupational radiation exposure
among radiographers?
"""


result = run_radsearch(question)


print("\n=== FINAL ANSWER ===")

print(result["messages"][-1].content)