# Docs review rubric

Use this when reviewing or rewriting the README and crate docs. Score each
category from 1 to 10. A change that raises the first eight must not lower
the last four: those protect things that earlier "improvements" removed.

## Craft

Benchmarked against Bevy, Avian, and `block-mesh`.

1. **Sentence structure**: clear, varied, no tangles.
2. **Style**: one consistent voice; no hype, no hedging.
3. **Readability**: scannable; jargon only where the reader needs it.
4. **User experience**: a reader can install, mesh, and migrate without
   guessing.
5. **Enjoyability**: the docs are pleasant to read.
6. **Flow**: sections arrive in the order a reader needs them.
7. **Engagement**: the opening makes the reader want to try it.
8. **Brevity**: nothing repeated; nothing `block-mesh` already documents.

## Guards

9. **Differentiation**: says plainly what this crate does differently,
   especially for AO. AO-safe meshing does not pack AO signatures into the
   meshing path. It preserves only the merge boundaries vertex AO depends
   on, derives them a whole row at a time with shifts and masks (the AO rule
   stays binary throughout), and leaves AO values to be computed once per
   final quad.
10. **Generosity**: credits cgerikj's binary greedy meshing and Inspirateur's
    Rust port warmly. Compare *techniques* and our own trade-offs; never
    list other projects' shortcomings, cite their issue trackers, or frame
    them as lacking.
11. **Evidence**: benchmarks keep the `visible_block_faces` column, because
    being competitive with or faster than the usual speed baseline is the
    strongest claim. Include a case where we lose, and state the hardware
    and settings.
12. **Author voice**: keep the author's own explanations and framings (for
    example "AO-Safe, Not AO-Packed" and "the AO rule stays binary the whole
    time") unless they are wrong. Rewrite for clarity, not for a generic
    tone.
