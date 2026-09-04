We will use IPEDS Trend Generator ("https://nces.ed.gov/ipeds/trendgenerator/") to create the evaluation ground truth dataset.

- Each question in the trend generator has an answer API and a dictionary API.
- The answer API returns the answer to the question for the latest year indexed by the key "answer".
- There is also a trend API that can be used to get the answer to the question for multiple years.

#### Answer Response

```json
{
    "answer": {},
    "expands": [{}],
    "filters": [{}]
}
```

`answer.question.questionText` contains the text of the question e.g. "What are the revenues (in thousands) of private not-for-profit postsecondary institutions using FASB standards?"

`answer` contains the answer to the question for the latest year \
`filter` contains the list of filters that can be applied to the question

```json
{
    [{
        questionId: 15,
        subjectId: 6,
        varId: 59
        variable: {
            varId: 59
            variableName: "HBCU"
            variableTitle,
            tableName,
            year,
            values: [{
            variableId:59
              valueCode: 1
              valueLabel: "Yes"
              ...
            },
            {
              variableId: 59
              valueCode: 2
              valueLabel: "No"
              ...
            }
            ]
        },
        year
    }]
}
```



#### GET Requests

ANSWER_URL = "https://nces.ed.gov/ipeds/trendgenerator/api/answer?subjectId={SUBJECT_ID}&questionId={QUESTION_ID}" \
DICTIONARY_URL = "https://nces.ed.gov/ipeds/trendgenerator/api/dictionary?subjectId={SUBJECT_ID}&questionId={QUESTION_ID}"

#### POST Requests

TREND_URL = "https://nces.ed.gov/ipeds/trendgenerator/api/trend" \
TREND_PAYLOAD: 
```json
{subjectId: "1", questionId: "1", filters: []}
```
TREND_FILTER_URL = "https://nces.ed.gov/ipeds/trendgenerator/api/trendTableRowOrColumn" \
TREND_FILTER_PAYLOAD:
```json
{
    subjectId: 6,
    questionId: 15,
    columnVarId: 59,
    columnVarValueCodes: ["1", "2"],
}
```

---

## The eval set

### Pipeline

```
fetch_raw.py     scrape + cache every API response      -> raw/questions.jsonl, raw/breakdowns.jsonl
build_dataset.py raw -> templated, stratified examples   -> dataset.jsonl
paraphrase.py    one natural rewrite per example         -> dataset.jsonl (in place)
run_eval.py      push to LangSmith, run the agent, score
```

`build_dataset.py` and everything after it are offline; only `fetch_raw.py` and
`paraphrase.py` touch a network.

### Two things that will bite you

**1. A trend row's number is the database table's year suffix; its label is not.**

`row.yearNumber` (or `row.year` in the plain trend response) maps straight onto the
table name — `2023` → `hd2023`, `effy2023`, `f2223_f1a`. The *label*
(`yearLabel`, or `columnValues[0]`) is the human data year and depends on the survey:
HD 2023 is labelled "2023-24", EFFY 2023 is labelled "2022-23". Questions must quote the
label; year filtering and DB coverage must use the number.

Our database stops at 2023 — there is no `hd2024`, no `f2324_*` — so the Trend
Generator's newest row is always dropped. "The latest year we have" therefore means
2023-24 for most questions but 2022-23 for the EFFY-backed ones.

**2. In a breakdown response, the `Total` column is not the total.**

`trendTableRowOrColumn` returns a column with `valueCode: ""` alongside the requested
codes. For an institution filter it is the real total, but for a content breakdown it
sums only the codes you asked for — request Asian and Black and the "Total" is 4.87M,
not the 24.9M of all students. Only per-`valueCode` values are used as ground truth.

### Variables

The same endpoint serves both lists on the answer response:

- `filters` — institution attributes (CONTROL, HBCU, ICLEVEL, FIPS, OBEREG, sector).
  Filtering by one tests the `hd<YYYY>` join and a valueset lookup.
- `expands` — everything else (gender, race, type of aid, source of funds, expense
  function, residency). These split the measure itself, so the agent has to find the
  right *column*, which is the harder and more realistic task.

`fetch_raw.py` treats any expand variable that isn't a known institution attribute as a
content breakdown.

### The unstated universe

Every published Trend Generator number is scoped to a universe no question mentions:

```sql
pset4flg IN (1,3) AND act IN ('A','N','R','M') AND sector <> 0 AND obereg <> 9
```

(Title IV institutions in the 50 states and DC; those four columns exist in every
`hd2004`…`hd2023`.) Without it the agent is wrong for a reason unrelated to its
reasoning, so `run_eval.py` appends it to the system prompt as `EVAL_SUFFIX`.
`backend/app/agent.py` is deliberately left untouched.

### Ground truth is reproducible

Checked against our own database at row year 2023, all exact:

| question | ground truth | our SQL |
|---|---|---|
| institutions eligible for federal aid | 5,686 (1878/1697/2111 by control) | `count(*)` on `hd2023` + universe |
| applications received | 13,901,237 | `sum(adm2023.applcn)` joined to `hd2023` |
| annual enrollment, HBCU = Yes | 343,518 | `effy2023` × `hd2023`, `effylev = 1` |
| annual enrollment, Asian / women | 1,698,425 / 14,528,758 | `efyasiat` / `efytotlw` |
