# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- `CONTEXT.md` at the repo root.
- `docs/adr/`, especially ADRs that touch the area about to be changed.

If any of these files do not exist, proceed silently. Do not flag their absence or suggest creating them upfront. Producer skills such as `/grill-with-docs` create them lazily when terms or decisions actually get resolved.

## File structure

Single-context repo:

```text
/
|-- CONTEXT.md
|-- docs/
|   `-- adr/
|       |-- 0001-example-decision.md
|       `-- 0002-example-follow-up.md
`-- src/
```

## Use the glossary's vocabulary

When your output names a domain concept in an issue title, refactor proposal, hypothesis, or test name, use the term as defined in `CONTEXT.md`. Do not drift to synonyms the glossary explicitly avoids.

If the concept you need is not in the glossary yet, either reconsider whether the project already uses different language, or note the gap for `/grill-with-docs`.

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding it.
