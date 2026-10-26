# Historical upstream methodology notes

The following material describes the upstream Qwen exploration. Its neuron IDs, dimensions,
measurements, and examples are historical observations, not evidence about the current Llama
checkpoint. Apply experimental principles cautiously and verify all model-specific claims
through the active explorer. The current agent prompt takes precedence over these notes.

# Interpreting neurons in Qwen3.5-4B: methodology

A working method for explaining a single MLP neuron, built from hands-on investigation in the explorer. It complements the explorer prompt above and uses its vocabulary (poles, steer lens, cone, web summary).

## Mindset

- **A neuron is a target, not a word list.** Every input makes a small judgement, and the strongest firing is where they all agree. Name the neuron from its strongest firings; weaker firings are usually the same idea seen from further away (Cammarata, *Neural Networks Are Already Clean*: neurons are analogizers, "Titanic-ish" rather than "Titanic facts").
- **Think in situations and distinctions.** Good labels describe a moment in text and what the neuron does there: "a line style comes next", "eggs as a counted ingredient", "we're comparing something, but HOW?". Noun categories ("hazard words", "graspable objects") are usually too literal.
- **Read the input and output as a pair.** "Already knows X → writes Y" is the most productive single move. `L20 N7531` is fed by `compare`, `either`, `comparison`, `corresponding` and writes `first/second`, `upper/lower`, `smaller/larger`: given that a comparison is happening, it supplies how to tell the items apart.
- **The strange part is the pointer.** When one view disagrees with the others, that disagreement is where the next test goes.
- **The network's ideas are often better than ours.** A clean neuron need not match an English word.

## Workflow

1. **Read `neuron` and `web_summary`.** Note `fed_by` / `held_by` (input), steer lens with KL (direct effect), `toward` / `away` and the strongest `writes_into` (downstream). Form a guess from the input/output pair.
2. **Steer on an open prompt** ("Tell me about your weekend.", "Write a short poem about the ocean.") at dose 0.6–0.8, then 1.5–2. Steering finds the axis faster than test sentences, especially for neurons that seldom fire. Read both rows: a meaningful negative pole writes something recognisable; a meaningless one just loops or breaks.
3. **Find where it fires hard.** Use `compare` with matched sets: target texts, near misses that share the vocabulary but not the situation, and distant controls. Raise the target until something exceeds the neuron's `act_max`. If nothing does, sweep 25–50 varied snippets (news, code, recipes, chat, fiction, maths, Chinese) with `compare` (24 texts per call) and read off the peaks.
4. **Check that the peak is real, then explain it from upstream.** At the peak token, zero candidate upstream neurons (`mode: mul, dose: 0`), whole MLP layers (`{layer, all: true, dose: 0}`) and attention per layer and per head, and measure the drop. Group the drivers by meaning.
5. **Break the hypothesis on purpose.** Write the test that would falsify it and run it: same frame with the key word swapped (prices vs customs differ; `def scale` vs `def process`), same word without the situation ("What color is bleach?").
6. **Run a blind maximum.** Write one sentence from the label alone. If it beats every example found so far, the label is near the center (`L20 N7531`: 2.44 from "The ratio of prices between the two regions differs by a factor of three", above its 0.93 dataset max).
7. **Close with logprobs.** `logprobs` with the neuron zeroed or pushed shows exactly which next tokens it moves and by how much. This is the final word on what the neuron does to the output.

## Reading each source

- **Steer lens.** Tokens such as `作为` (as a…), `您的` (your, polite), `无法` / `cannot`, `我是一个` (I am a…) appear across a great many neurons. They are a signature of measuring on chat prompts and carry little information about the specific neuron. Weight the remaining, specific tokens.
- **Logit lens.** Most useful in late layers. Specific repeated tokens (`method` × 12, `egg` × 12) are reliable; mixed fragments are noise.
- **Web summary.** Pooled steer-lens words over the cone. A token on both `fed_by` and `held_by` (e.g. `XSS`, `（`, `.`) often reflects pooling; treat shared topical words differently, see "Fed and held by the same topic" below.
- **Weight links (`writes_into`, `reads_from`, cone).** Leads to check, never circuits. Verify any claimed path with activations under ablation; `L22 N3467 → L31 N1348` was a real, causal link whose removal changed nothing in the output.
- **Activation range.** Raw values. Scores many times `act_max` (18× for cosine, 39× for line styles) mark a detector at its center.

## Motifs seen in this model

- **Lookahead.** Many neurons fire on the token before the thing they predict: `L25 N4719−` peaks at "r times **the**" before "cosine" (−15) and at "sin² **+**" before "cos²"; `L22 N5826` peaks at "while **the**" before "dashed".
- **Summary positions.** Mood and planning neurons peak on commas, periods, `?`, `</think>` and the blank line before a reply, where attention gathers the text so far. `L22 N3467` (critical warning) reaches 6.2 at `</think>` before an answer to a bleach question.
- **Reads one way, writes the other.** Some neurons fire positive on a topic while their positive direction writes the opposite register: `L18 N8827` fires on lurid content and, pushed positive, makes text gentle and serene; its negative pole is obscenity.
- **Fed and held by the same topic.** A neuron that draws a distinction inside a topic is wired to neurons on both sides of it. `L10 N6032` is fed by `unsafe`, `vulnerable`, `attack` and held by `secure`, `Police`, `FDA`: it marks danger with no safeguard in place.
- **One direction, signed by context (gain control).** `L23 N7748` adds eggs in counted-ingredient slots (+5.75 at "a dozen") and, at −3.3 in "bacon and fried", holds eggs back so `potatoes` and `toast` stay in play. Logprobs confirm both.
- **Distributed triggers.** Planning neurons are often driven by many small inputs plus attention rather than one upstream switch. `L22 N3467`'s 6.2 is spread across L9–L10 MLPs, L17–L18 heads and several safety neurons, with L11–L17 MLPs acting as a brake.
- **High-gain, rare writers.** A huge steer KL with almost no natural firing marks a lever whose trigger lies outside test sentences. `L26 N4405` (KL 7.4) turns any answer into zeros ("A spider has 0 legs") yet fires only weakly on CSS `margin`, with negligible effect there. Label these by effect and mark the trigger as unknown.
- **Meaning over spelling.** `L20 N7531` fires on `is_larger`, `compare`, `ratio`, `differ` but not on `>`, `==` or `max()`; `L25 N4719` ignores words that merely start with "co-".

## Doses

- 0.5–0.8 shows the direction while keeping text coherent; 1.5–2 often collapses into loops. Last-layer token neurons (e.g. `L29 N1053`, "bro") loop at any dose.
- Pushes add up across neurons; when stacking, scale each one down.

## Writing it up

- Lead with the strongest firings and the steering result, then the near misses that bound the label.
- State the verdict (one job / two related jobs / not yet understood) with confidence, and say which view you weight when views disagree.
- Name what remains unexplained.
- End with the specific test that would break the explanation.

## Reference neurons

- `L10 N6032+`: risky how-to request ("How do I store gasoline?") → caution voice (`我必须` I must, `我无法` I can't).
- `L11 N6261+`: refusal lever; +1.5 moves the first reply token from "I" (99.6%) to "Sure / Yes / Absolutely".
- `L12 N699`: + AI self-limitation ("I can only…"), − bubbly persona ("What a fun day!").
- `L18 N6311`: + frame as a method ("the sourdough method… why this works"), − pontificate about the topic.
- `L18 N8737+`: sexual desire → euphemistic bedroom talk.
- `L19 N7839−`: conspiracy / "the truth they don't want you to know" (−12.6), weak as a steering lever.
- `L19 N9066+`: felt panic (7.3 on exam panic), writes anxious rambling at +0.8.
- `L20 N7531`: comparing amounts → which item / how they rank.
- `L22 N3467`: plans a ⚠️ critical warning before answering a hazardous request.
- `L22 N5826`: a line style comes next (dashed / dotted / broken).
- `L23 N7748`: writes eggs; + counted-ingredient slots, − keeps other foods in play.
- `L25 N4719−`: cosine comes next (x-component → cos).
- `L26 N4405`: writes zero; trigger unknown.
