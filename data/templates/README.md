# Syntax templates

Each `.jsonl` file contains one JSON object per line:

```json
{"syntax_id":"wh_adverb_fronted","template":"Where was {person} born?"}
```

Files and answer types:

- `born_in_city.jsonl` → town/city
- `lives_in_city.jsonl` → town/city
- `born_in_date.jsonl` → ISO date
- `profession.jsonl` → profession

Replace `{person}` with a value from `data/strange_names.txt`.

The templates intentionally use canonical relation vocabulary:

- birth location: `born`, `city`, `city of birth`
- residence: `live`, `city`, `city of residence`
- birth date: `born`, `date`, `date of birth`
- profession: `work`, `profession`

Templates differ by syntax rather than by synonym choice. Their `syntax_id` values describe the main construction: wh-fronting, preposition stranding, an in-situ wh-phrase, a possessive noun phrase, a relative clause, a cleft, an embedded question, a topic-comment construction, or a fragment.

`data/synonyms.json` is kept separate and is not applied to these templates yet. For controlled experiments, split examples by `syntax_id` before adding lexical variants.
