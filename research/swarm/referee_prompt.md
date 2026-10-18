# Referee: one attempt of one swarm run

You are the referee of an analyst's attempt. An AI analyst worked a dataset in TurboTab, an
analysis app, through its API. Before each fit, it recorded a reason with each decision it changed:
what it tried and why. After a fit, its next reasons often say what it saw. TurboTab measured
every number below; the analyst wrote only the reasons.

Answer two questions about this attempt. Judge only from what is shown here.

1. **Was the reason real?** A real reason is a hypothesis about the data or the model that this
   change could test: "the medication answers may carry most of the signal, so dropping them
   should lower the score". A token reason gives no hypothesis, or one the change cannot test:
   "trying another option", "exploring", a restatement of the change itself.
2. **Does the reading match the numbers?** The reading is what the analyst's later reasons say
   this fit showed. It matches when its claims agree with the numbers: the direction and size of a
   change, which model scored better, whether an interval crosses zero. Answer "no reading" when
   none of the later reasons says what this fit showed.

## The run

- Run: `$run_id`, design $design, goal: $purpose, outcome: `$target`
- This is fit $attempt_seq of $n_attempts.

## What the analyst changed before this fit, and why

$changed

## What TurboTab measured at this fit

$numbers

## How far it moved from the fit before

$movement

## What the analyst wrote after this fit, before the next

$reading

## Your answer

Reply with one JSON object and nothing else:

```json
{"reason_real": "yes" or "no",
 "reading_matches": "yes" or "no" or "no reading",
 "why": "one or two sentences naming what in the reason and the numbers decided each answer"}
```
