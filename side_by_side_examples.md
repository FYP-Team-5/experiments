# Base vs feedback adapter in the comparison pipeline: side-by-side examples

The rows of the feedback repo's side-by-side examples (`runs/side_by_side_examples.md` there), plus two more on the same questions, from the third comparison run (`cmp_feedback_base` and `cmp_feedback_adapter`, seed-42 test split). Two things differ from the feedback repo:

- The feedback is written for the **QLoRA grader's marks**, not the dataset's. The key points below are the grader's; where the dataset disagrees, it is shown in brackets.
- Each feedback is checked by the **9B verifier**, which can reject it and ask for a rewrite, up to two times. A draft rejected after the second rewrite is kept anyway. Only the first and the last draft are saved.

Judge verdicts are from the Qwen3.5-27B judge (`judge_feedback_27b.ipynb`, results in [results/comparison/judge_27b/](results/comparison/judge_27b/)), against the marks the feedback was written for. Feedback is quoted verbatim; Markdown characters such as `**` and backticks are escaped, so they show exactly as the model wrote them.

| 387 rows | Base | Adapter | Base after rewrites | Adapter after rewrites |
|---|---|---|---|---|
| Passes all four judge checks | 81.9% | 90.4% | 82.7% | 89.7% |
| Faithful to the student's answer | 89.4% | 93.3% | 88.9% | 92.8% |
| Factually correct | 89.7% | 95.9% | 91.2% | 95.1% |
| Rows where only this model passed | 15 | 48 | 16 | 43 |

The verifier rewrote 165 base and 106 adapter feedbacks. Its rewrites turned 25 base fails into passes and 22 passes into fails; for the adapter, 6 and 9. That is why they leave the pass rates almost unchanged.

Examples 1–4 are rows the base model failed at first. In 1–2 the verifier catches the error and the rewrite fixes it; in 3–4 the grader's marks are wrong and the feedback follows them. Examples 5–6 passed for both and show the difference in style. Examples 7–8 are rows where the adapter failed and the base model passed. Examples 9–10 show rewrites making feedback worse.

## 1. Base credits the student with reasoning they never wrote; the verifier catches it
Row 3050 · partial

**Question:** Q1. State TRUE or FALSE and justify. No correct justification, no marks.
e. If a parent process forks a new child process, the exec system call cannot be used in the parent process until the child process terminates.

**Reference answer:** False, parent and child processes are two independent processes. Parent can perform any operation that it likes including an exec system call.

**Student answer:** \*\*FALSE\*\*

**Key points:**
- MET: Answer states the statement is False
- NOT MET: Answer mentions that parent and child are independent processes
- NOT MET: Answer mentions that the parent can call exec even before the child terminates

**Base Qwen3.5-9B, first draft** (87 words). Judge: failed: Feedback claims student wrote justification text that is absent from their answer.

*9B verifier:* rejected: “The feedback praises the student for noting that 'parent and child processes are independent,' yet the rubric explicitly marks this criterion as NOT MET because the student's answer contained only the word 'FALSE' without any justification text.”

> You correctly identified the statement as \*\*FALSE\*\*, which is a great start! However, your justification was incomplete. While you noted that parent and child processes are independent, you didn't explicitly state that the parent retains the freedom to execute the \`exec\` system call immediately, regardless of whether the child has terminated. To fully justify your answer, make sure to connect the independence of the processes directly to the parent's ability to perform operations like \`exec\` at any time. Keep up the good work on spotting the error!

**Base Qwen3.5-9B, after 1 rewrite** (61 words). Judge: passed.

*9B verifier:* accepted

> Your answer correctly identifies the statement as \*\*FALSE\*\*, which is the only criterion met. However, you did not provide any justification text, which is required to earn full marks. To improve, explicitly explain that parent and child processes are independent entities, meaning the parent can execute the \`exec\` system call immediately after forking, regardless of whether the child process has terminated.

**Feedback adapter** (99 words). Judge: passed.

*9B verifier:* accepted

> This question tests whether you understand that a parent process remains independent of its child after forking. You correctly identified the statement as false, but your justification was incomplete because it did not explain that the parent and child are separate entities. Specifically, you need to clarify that the parent can execute an exec system call at any time, regardless of whether the child process is still running. A complete answer would state that the claim is false because parent and child processes operate independently, allowing the parent to call exec immediately without waiting for the child to terminate.

*What to notice:* The base model's first draft makes the same mistake as in the feedback repo: "you noted that parent and child processes are independent", when the student wrote only "FALSE". The 9B verifier rejects it for exactly that reason, and the rewrite ("you did not provide any justification") passes the judge. The adapter gets it right the first time.

## 2. Base confirms a wrong answer; the verifier catches it
Row 890 · none met

**Question:** What does the address (&) operator return?

**Reference answer:** The memory address of its operand.

**Student answer:** the reference to a variable

**Key points:**
- NOT MET: Answer mentions returns memory address of operand

**Base Qwen3.5-9B, first draft** (68 words). Judge: failed: Feedback incorrectly states the & operator returns a reference, which is factually inaccurate.

*9B verifier:* rejected: “The feedback praises the student's answer ('You are very close!') and validates the concept of a 'reference,' whereas the rubric explicitly requires the term 'memory address' and the score is 0 because the student failed to use this specific terminology.”

> You are very close! The operator does indeed return a reference to a variable, but in C/C++, the specific term used is "memory address." While a reference is conceptually similar, the \`&\` operator literally yields the numerical address where the operand is stored in memory. Try adjusting your answer to explicitly state that it returns the \*\*memory address\*\* of the operand to match the precise definition. Great start!

**Base Qwen3.5-9B, after 1 rewrite** (52 words). Judge: passed.

*9B verifier:* accepted

> Your answer describes a "reference," which is a distinct concept from what the \`&\` operator actually does. The \`&\` operator does not return a reference; instead, it returns the specific \*\*memory address\*\* of its operand. To improve your answer, replace the word "reference" with "memory address" to accurately reflect the operator's function.

**Feedback adapter** (52 words). Judge: passed.

*9B verifier:* accepted

> The address-of operator specifically returns the memory address of its operand, not a general reference to a variable. While your answer captures the idea of pointing to a variable, it lacks the precise terminology required for this context. To improve, explicitly state that the operator yields the memory address of the operand.

*What to notice:* Again the base model tells the student the operator "does indeed return a reference" and that they are "very close". The verifier rejects it, and the rewrite says plainly that `&` does not return a reference. The adapter needs no rewrite.

## 3. The grader's wrong mark carries through
Row 1850 · all met (the dataset's marks: none met)

**Question:** What is the main difference between a string of characters that is read into a variable of type string versus a variable of type char[]?

**Reference answer:** The char[] will automatically add a null \\0 character at the end of the string.

**Student answer:** the one that is type string has an extra string member. The /0 at the end.

**Key points:**
- MET (dataset: NOT MET): Answer mentions char[] adds null terminator

**Base Qwen3.5-9B** (48 words). Judge: failed: Feedback claims student said char[] adds null, but student said string has it.

*9B verifier:* accepted

> Great job! You correctly identified that the \`char[]\` type automatically appends the null terminator (\`\\0\`) at the end of the string. This is the crucial distinction, as \`String\` objects in Java manage their own memory and do not require this manual termination. Your answer is accurate and complete.

**Feedback adapter** (57 words). Judge: failed: Feedback claims student identified char[] has null terminator, but student wrote string.

*9B verifier:* accepted

> Your answer correctly identifies that the char[] type includes a null terminator at the end of the string. You accurately noted this specific difference, which is the key distinction between how these two types handle character data in memory. This response works because it directly addresses the automatic addition of the null character required for C-style strings.

*What to notice:* The student said `string` has the `\0`, which is wrong, but the QLoRA grader marked the point MET. Both models then praise the student for saying `char[]` adds the null terminator, the opposite of what they wrote, and the base model brings in Java's `String`. The judge fails both even against the grader's marks, and the 9B verifier accepts both. In the feedback repo, written for the dataset's NOT MET, the adapter passed this row.

## 4. A wrong grade the rewrites can't fix
Row 372 · all met (the dataset's marks: none met)

**Question:** What is the role of a header-file?

**Reference answer:** To store a class interface, including data members and member function prototypes.

**Student answer:** \#include "hat.h"  allows the use of the hat container, so called from the act of pulling names from a hat.  The header file for hat includes functions for storage and retrieval of it's contents, mainly put(), peek(), and pull().

**Key points:**
- MET (dataset: NOT MET): Answer mentions storing class interface
- MET (dataset: NOT MET): Answer mentions data members and member function prototypes

**Base Qwen3.5-9B, first draft** (91 words). Judge: failed: Feedback contradicts MET status by suggesting data members were missing.

*9B verifier:* rejected: “The feedback incorrectly suggests the student missed the 'data members' criterion, contradicting the assigned score where both criteria (including data members) are marked as MET.”

> Great job! You correctly identified that the header file serves to expose the class interface, specifically by listing the member functions like \`put()\`, \`peek()\`, and \`pull()\`. You also accurately noted that this allows other files to use the class.
>
> To make your answer even more complete, try explicitly mentioning that the header file also stores the \*\*data members\*\* (the variables) of the class, not just the function prototypes. Including both the data and the function signatures will fully capture the role of a header file. Excellent work on the specific examples!

*The draft in between was not saved. 9B verifier on it:* rejected: “The feedback claims the student correctly identified that the header file contains details for 'data members,' but the student answer only lists specific member function prototypes (put(), peek(), pull()) and never mentions data members.”

**Base Qwen3.5-9B, after 2 rewrites** (68 words). Judge: failed: Feedback contradicts the marker's decision by claiming a MET point is missing.

*9B verifier:* rejected: “The feedback incorrectly claims the student's answer is incomplete regarding data members, despite the rubric explicitly marking the criterion 'Answer mentions data members' as MET and the assigned score reflecting full marks.”

> Great job identifying that the header file stores the class interface and listing specific member function prototypes like \`put()\`, \`peek()\`, and \`pull()\`. However, your answer currently only describes the functions. To fully capture the role of a header file, you should also mention that it declares the \*\*data members\*\* (the variables) associated with the class, not just the functions. Adding this detail will ensure your explanation is complete.

**Feedback adapter** (60 words). Judge: failed: Feedback claims student mentioned 'data members', but their answer only lists functions.

*9B verifier:* accepted

> Your answer correctly identifies that a header file stores the class interface, including data members and function prototypes like put(), peek(), and pull(). You effectively illustrated this by describing how the hat container uses these declarations to manage its contents. This response works because it clearly demonstrates the specific role of a header file in defining a class's public interface.

*What to notice:* The grader marked both points MET; the dataset marks both NOT MET, and the student never mentions data members or prototypes. Judging by the verifier's notes, the base model's drafts swing between telling the student data members are missing (which contradicts the MET mark) and crediting the student with them (which isn't in the answer). The 9B verifier rejects all three drafts and the last one is kept anyway. The adapter credits the student with "data members" and the verifier accepts it. The marks leave the feedback little room: it may neither say data members are missing nor credit the student with them.

## 5. Both pass: style on a fully correct answer
Row 2172 · all met

**Question:** Q1. State TRUE or FALSE and justify. No correct justification, no marks.
e. If a parent process forks a new child process, the exec system call cannot be used in the parent process until the child process terminates.

**Reference answer:** False, parent and child processes are two independent processes. Parent can perform any operation that it likes including an exec system call.

**Student answer:** \*\*False\*\*, after fork is executed the two processes are completely independent of each other and so the parent can call exec system call whenever it wants.

**Key points:**
- MET: Answer states the statement is False
- MET: Answer mentions that parent and child are independent processes
- MET: Answer mentions that the parent can call exec even before the child terminates

**Base Qwen3.5-9B** (87 words). Judge: passed.

*9B verifier:* accepted

> Excellent work! You correctly identified the statement as \*\*False\*\* and provided a precise justification.
>
> Your explanation that the parent and child processes are "completely independent" is spot on. This independence is exactly why the parent process retains full control and can execute the \`exec\` system call immediately, regardless of whether the child process has finished running.
>
> You have perfectly captured the core concept: the parent is free to perform any operation, including replacing its own image with a new program, without waiting for the child. Great job!

**Feedback adapter** (62 words). Judge: passed.

*9B verifier:* accepted

> This question tests your understanding of process independence after a fork system call. You correctly identified the statement as false and explained that the parent and child become completely independent, allowing the parent to execute an exec call at any time. Your answer works because it clearly establishes that the termination of the child is not a prerequisite for the parent's execution.

*What to notice:* Both are correct and neither is rewritten. As in the feedback repo, the base model uses bold text, three paragraphs and generic praise; the adapter writes one plain paragraph, word for word the same as there.

## 6. Both pass: style on a one-word answer
Row 88 · all met

**Question:** Where do C++ programs begin to execute?

**Reference answer:** At the main function.

**Student answer:** main method

**Key points:**
- MET: Answer mentions main function

**Base Qwen3.5-9B** (55 words). Judge: passed.

*9B verifier:* accepted

> Great job! You correctly identified that C++ programs start execution in the \`main\` function. Just a small note: in C++, we refer to it as the \`main\` \*\*function\*\*, whereas "method" is typically used for Java or C\#. Using the precise terminology will help you communicate more accurately in the future. Keep up the excellent work!

**Feedback adapter** (47 words). Judge: passed.

*9B verifier:* accepted

> Your answer correctly identifies that C++ programs begin execution at the main function. While you used the term "method," which is more common in Java, the core concept here is accurate for C++. This response works because it directly names the specific function where program execution starts.

*What to notice:* Both point out that "method" is Java terminology. The base model uses backticks and bold and closes with "Keep up the excellent work!"; the adapter stays plain. The adapter's text is the same as in the feedback repo.

# Where the adapter is worse

## 7. Adapter contradicts the marker; the verifier misses it
Row 362 · partial

**Question:** What is the role of a header-file?

**Reference answer:** To store a class interface, including data members and member function prototypes.

**Student answer:** Header files separate files in which classes are defined and allow the compiler to recognize classes when used elsewhere.

**Key points:**
- MET: Answer mentions storing class interface
- NOT MET: Answer mentions data members and member function prototypes

**Base Qwen3.5-9B** (66 words). Judge: passed.

*9B verifier:* accepted

> You correctly identified that header files separate class definitions and enable the compiler to recognize them in other files, which is a great start. To fully capture the role of a header file, be sure to specify that it stores the \*\*interface\*\*, which explicitly includes \*\*data members\*\* and \*\*member function prototypes\*\*. Adding these specific details will complete your explanation of what the header file actually contains.

**Feedback adapter** (58 words). Judge: failed: Feedback contradicts MET Point 1 by claiming student did not specify storing class interface.

*9B verifier:* accepted

> Your answer correctly identifies that header files separate class definitions and help the compiler recognize them. However, you did not specify that these files store the class interface, which includes both data members and member function prototypes. A complete answer would state that a header file stores a class interface, including its data members and member function prototypes.

*What to notice:* The marker marked "storing class interface" as MET, and the adapter again says "you did not specify that these files store the class interface". The 9B verifier accepts it, so it is never rewritten. The base model gets this row right.

## 8. Adapter confirms a wrong claim; the verifier catches it
Row 480 · none met

**Question:** What is the main advantage associated with function arguments that are passed by reference?

**Reference answer:** It avoids making copies of large data structures when calling functions.

**Student answer:** You do not alter the original value of the variable that was passed.

**Key points:**
- NOT MET: Answer mentions avoiding copies of large data structures

**Base Qwen3.5-9B** (70 words). Judge: passed.

*9B verifier:* accepted

> Your answer highlights an important distinction: passing by reference does not alter the original variable if you are only reading its value. However, the main advantage you are looking for relates to memory efficiency. When arguments are passed by reference, the program avoids creating duplicate copies of large data structures, which saves significant memory and improves performance. Try focusing your response on how this approach prevents unnecessary duplication of data.

**Feedback adapter, first draft** (59 words). Judge: failed: unreadable (the judge ran out of tokens while thinking).

*9B verifier:* rejected: “The feedback explicitly validates the student's answer as 'correct' regarding the behavior of pass-by-value, yet the assigned score is 0 because the specific rubric criterion about avoiding copies of large data structures was not met.”

> The main advantage of passing arguments by reference is that it avoids creating copies of large data structures when calling functions. Your answer correctly notes that the original variable's value remains unchanged, which describes a behavior of pass-by-value or const references rather than the primary efficiency benefit of pass-by-reference. How does avoiding memory duplication improve performance for large datasets?

**Feedback adapter, after 1 rewrite** (61 words). Judge: passed.

*9B verifier:* accepted

> Your answer describes the behavior of pass-by-value, where the original variable remains unchanged, but it does not address the specific advantage of pass-by-reference. The main benefit is that passing by reference avoids creating copies of large data structures when calling functions, which saves memory and improves performance. How does avoiding these copies help when working with very large arrays or objects?

*What to notice:* The adapter's first draft is the one the feedback repo's judge failed: it says the answer "correctly notes" that the value is unchanged. Here the 27B judge ran out of tokens on it, which also counts as a fail. The 9B verifier rejects it, and the rewrite says the student described pass-by-value; it passes. The base model passes without a rewrite.

# Where the rewrites make it worse

## 9. A rewrite adds a factual error
Row 386 · all met

**Question:** What is the role of a header-file?

**Reference answer:** To store a class interface, including data members and member function prototypes.

**Student answer:** A header file usually contains class and/or function prototypes.

**Key points:**
- MET: Answer mentions storing class interface
- MET: Answer mentions data members and member function prototypes

**Base Qwen3.5-9B, first draft** (33 words). Judge: passed.

*9B verifier:* rejected: “The feedback praises the student for mentioning 'data members', but the student answer only states 'class and/or function prototypes' and completely omits any mention of data members, contradicting the assigned score of 2/2.”

> Great job! You correctly identified that header files store class and function prototypes, which effectively covers the concept of a class interface. Your answer is clear and accurate. Keep up the excellent work!

*The draft in between was not saved. 9B verifier on it:* rejected: “The feedback incorrectly suggests the student missed the 'data members' criterion, contradicting the assigned score which marks this criterion as MET.”

**Base Qwen3.5-9B, after 2 rewrites** (43 words). Judge: failed: Feedback claims headers store function definitions instead of prototypes.

*9B verifier:* rejected: “The feedback claims the student implicitly covered 'data members' because the class interface encompasses them, but the student's answer only mentioned 'class and/or function prototypes' and never referenced data members.”

> Excellent work! You correctly identified that header files store class interfaces and function prototypes. By mentioning "class prototypes," you implicitly covered the inclusion of data members, as the class interface encompasses both its data and function definitions. Your answer is complete and accurate.

**Feedback adapter, first draft** (60 words). Judge: failed: Feedback states student identified 'data members', a term absent from their answer.

*9B verifier:* rejected: “The feedback claims the student mentioned 'data members', but the student answer only stated 'class and/or function prototypes' without explicitly referencing data members.”

> Your answer captures the essential purpose of a header file by noting that it holds class and function prototypes. This correctly identifies how these files store the interface, including data members and function signatures, which is exactly what the question asked for. Your response works because it accurately describes the core components found in a header file without unnecessary detail.

*The draft in between was not saved. 9B verifier on it:* rejected: “The feedback incorrectly claims the student failed to mention data members, contradicting the assigned score which marks the criterion 'Answer mentions data members and member function prototypes' as MET.”

**Feedback adapter, after 2 rewrites** (46 words). Judge: passed.

*9B verifier:* rejected: “The feedback praises the student for covering 'data members' and 'member function prototypes,' but the student answer only mentions 'class and/or function prototypes,' completely omitting the specific requirement for 'data members' which is marked as MET in the criteria.”

> Your answer correctly identifies that header files store class and function prototypes, which effectively covers the required interface details. This response works because it accurately describes the primary purpose of a header file in defining the structure and methods available to other parts of a program.

*What to notice:* The base model's first draft passes. The 9B verifier rejects it for praising "data members", which it never mentions, and rejects both rewrites too. The kept draft says the class interface includes "function definitions", the same factual error the base model made on this question in the feedback repo, and the judge fails it. On the same row the verifier helps the adapter: its first draft credits the student with data members, and the final draft drops that and passes the judge, though the verifier still rejected it.

## 10. A rewrite removes a correct correction
Row 1834 · all met

**Question:** What is the main difference between a string of characters that is read into a variable of type string versus a variable of type char[]?

**Reference answer:** The char[] will automatically add a null \\0 character at the end of the string.

**Student answer:** the type char[] has a  null ('\\n' ) element at the very end.

**Key points:**
- MET: Answer mentions char[] adds null terminator

**Base Qwen3.5-9B, first draft** (60 words). Judge: passed.

*9B verifier:* rejected: “The feedback incorrectly treats the student's error (confusing '\\n' with '\\0') as a minor precision issue while assigning full marks, despite the rubric explicitly requiring the mention of the null terminator '\\0' for the point to be awarded.”

> Great job! You correctly identified that a \`char[]\` automatically includes a null terminator at the end, which is the key distinction from a \`String\` object. While you mentioned \`'\\n'\` (newline), the specific null character added is \`'\\0'\`. Just remember to use \`'\\0'\` instead of \`'\\n'\` in your explanation to be precise. Otherwise, your understanding of the difference is spot on!

*The draft in between was not saved. 9B verifier on it:* rejected: “The feedback incorrectly claims the student's answer is wrong regarding the null terminator, yet the system assigned full marks (1/1) because the criterion 'Answer mentions char[] adds null terminator' was marked MET, ignoring the student's specific error of calling it '\\n'.”

**Base Qwen3.5-9B, after 2 rewrites** (71 words). Judge: passed.

*9B verifier:* unreadable verdict, which counts as a rejection

> You correctly identified that a \`char[]\` automatically adds a null terminator at the end of the string, which is the key difference from a \`String\` object. However, be careful with the specific character you mentioned: the null terminator is the null character (\`\\0\`), not a newline (\`\\n\`). While both are invisible, they serve different purposes, so it's important to distinguish them in your explanation. Great job on catching the core concept!

**Feedback adapter, first draft** (73 words). Judge: passed.

*9B verifier:* rejected: “The feedback incorrectly claims the student identified the null terminator as '\\n' (newline), whereas the student's answer explicitly stated '\\n' as the null element, which is factually wrong and contradicts the reference answer that specifies '\\0'.”

> Your answer correctly identifies that a char[] includes a null element at the end, which is the key distinction from a string variable. However, it is important to note that this terminator is the character '\\0', not the newline character '\\n'. This specific detail ensures the system knows where the text ends. When you see a question about memory layout, can you recall exactly which byte marks the boundary of a C-style string?

*The draft in between was not saved. 9B verifier on it:* rejected: “The feedback incorrectly treats the student's error (confusing null terminator with newline) as a reason to withhold the full mark, despite the rubric and assigned score indicating the criterion 'char[] adds null terminator' is fully MET.”

**Feedback adapter, after 2 rewrites** (53 words). Judge: failed: Feedback validates student's incorrect symbol ('\\n') as the accurate null terminator.

*9B verifier:* rejected: “The feedback incorrectly validates the student's claim that the null terminator is '\\n' (newline), whereas the student explicitly stated '\\n' instead of the correct null character '\\0'.”

> Your answer correctly identifies that a char[] variable includes a null terminator at the end of the string. You accurately noted this element is present, which is the key distinction from a standard string type. This response works because it directly addresses the automatic addition of the null character required for C-style strings.

*What to notice:* The student called the null terminator `'\n'`, but the grader marked the point MET. The adapter's first draft accepts the point and corrects `\n` to `\0`, and it passes. The 9B verifier rejects it for a muddled reason, then rejects the next draft for treating the error "as a reason to withhold the full mark". The final draft drops the correction and says the student "accurately noted" the terminator; the judge fails it. The base model's rewrites keep the correction, and it passes.
