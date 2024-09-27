WHETHER_DECOMPOSE = 'Question: {question}\nDecide if the question can be directly answered, or the question should be decomposed into sub-questions for easier answering. If the question can be directly answered, please answer \"Yes.\" If the question should be decomposed for easier answering, please answer \"No.\"'

DIRECT_AOKVQA = 'Please answer the following question, your answer should mention both the option letter and the word:\n{question}'
DIRECT_GENERAL = 'Please answer the following question:\n{question}'

DECOMPOSE_FIRST = "Question: {question}\nPlease firstly decompose the given question into several image-relevant sub-questions to help you answer the given question. Please avoid giving repeated subquestions or generating an excessive number. Feel free to suggest an appropriate quantity based on your judgment."
DECOMPOSE_SECOND = 'Please answer each of the sub-questions raised by yourself in the previous step.'
DECOMPOSE_THIRD_AOKVQA = 'With the help of the already answered sub-questions, please answer the original question, your should mention both the option letter and the word:\n{question}'
DECOMPOSE_THIRD_GENERAL = "With the help of the already answered sub-questions, please answer the original question:\n{question}"

PROMPT_DICT = {'whether_decompose': WHETHER_DECOMPOSE, 'direct_aokvqa': DIRECT_AOKVQA, 'direct_general': DIRECT_GENERAL, 'decompose_first': DECOMPOSE_FIRST, 'decompose_second': DECOMPOSE_SECOND, 'decompose_third_aokvqa': DECOMPOSE_THIRD_AOKVQA, 'decompose_third_general': DECOMPOSE_THIRD_GENERAL}