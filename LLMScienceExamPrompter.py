from axolotl.prompters import AlpacaPrompter

class LLMScienceExamPrompter(AlpacaPrompter):

    system_prompt = """You are a genius bot that always correctly answer a multiple choice question with options [A, B, C, D, E] base on knowledge if provided. Please determine whether the background knowledge is useful first, if it is useful, please answer the question based on the knowledge, otherwise, please answer the question and ignore the knowledge. **You are only allowed to reply in the following format, for example, "useful:True||though:Because...||answer:A" . The "useful" is a boolean value, and "thought" is your reasoning thought about the question and the answer which should be one of [A, B, C, D, E].**\n"""
    system_no_input_prompt = "nono"


if __name__ == "__main__":
    knowledge = """<rmpty>"""
    
    example = dict(
        instruction = f"Please answer the quaetion base on the the knowledge:{knowledge}",
        input = """Which of the following statements accurately describes the impact of Modified Newtonian Dynamics (MOND) on the observed "missing baryonic mass" discrepancy in galaxy clusters?\nA MOND is a theory that reduces the observed missing baryonic mass in galaxy clusters by postulating the existence of a new form of matter called "fuzzy dark matter."\nB MOND is a theory that increases the discrepancy between the observed missing baryonic mass in galaxy clusters and the measured velocity dispersions from a factor of around 10 to a factor of about 20.\nC MOND is a theory that explains the missing baryonic mass in galaxy clusters that was previously considered dark matter by demonstrating that the mass is in the form of neutrinos and axions.\nD MOND is a theory that reduces the discrepancy between the observed missing baryonic mass in galaxy clusters and the measured velocity dispersions from a factor of around 10 to a factor of about 2.\nE MOND is a theory that eliminates the observed missing baryonic mass in galaxy clusters by imposing a new mathematical formulation of gravity that does not require the existence of dark matter.""",
        output = """useful:True||though:Because...||answer:A### END."""
    )
    
    
    print("="*100)
    prompter = LLMScienceExamPrompter()
    example = prompter.build_prompt(**example)
    print(''.join(example))
    print("="*100)
