---
name: terse-comments
description: Write or tighten a comment or docstring in this repo - state the decision, then only what the code cannot say. Use when writing a new comment or docstring, when asked to make one shorter or less wordy, or when one argues its case instead of stating it.
---

# terse-comments

Expect to cut a third to a half of what you first write. That ratio is a smell, not a target. A comment already made of distinct facts is finished. Cutting to hit a number removes a fact.

## First: is it agent-facing?

A docstring on an agent tool is the model's specification. Any string that reaches a prompt is also a specification. Apply sentence form only. Do not delete redundant text. A tool docstring that restates an argument does its job. The options, examples, and limits define behavior. If shortening drops one, keep the long sentence.

The remaining rules apply to text for humans. If the audience is unclear, treat it as agent-facing.

## Docstrings

**Paragraph 1 is the contract.** State what the item returns or does. Use one sentence when possible. Then state any intended behavior that can look like a bug.

**Paragraph 2 contains facts that code cannot show.** Include external API behavior, rejected alternatives, and consequences with no local symptom.

Do not add a third paragraph. Merge a necessary fact into the second paragraph.

## Comments

A comment has no contract paragraph because it cannot restate a signature. Lead with the decision. Then give the reason if it is not clear.

Code follows the comment, so delete narration. Also delete a comment when the reader sees its consequence in the next lines.

## Cut

- **The justification chain.** "This is intended behavior" can replace the scenario that proves it.
- **The closing flourish.** Delete the last sentence when it only reframes the paragraph.
- **The contrast clause.** Delete a trailing contrast when the claim stands without it.
- **The pointer that restates.** Use a pointer or the repeated reason, not both.
- **Vendor mechanism trivia.** Name the constraint, not the vendor feature matrix.
- **Orientation.** Do not explain locations, module transport, or another file's argument when the repository shows them.
- **Advice, down to its imperative.** State the constraint without a worked recommendation.

## Keep

- External behavior that requires a test to learn.
- The reason an obvious call was not used and the failure it causes.
- A consequence that the local code does not show.
- The security reason for a boundary.
- Exact contract terms such as half-open ranges, ordering, units, and pairings.
- A repeated idiom at every site when repetition makes the pattern clear.

## Two places say the same thing

Keep the statement at the higher level and delete the other. A module fact belongs in the module docstring.

Check both copies before you delete either one. This mistake loses information instead of only damaging style.

## Notes

Shorter text can beat ideal grammar. Passive voice is acceptable when it uses fewer words.

Verify a claim before you compress it. Shortening a claim asserts it again.
