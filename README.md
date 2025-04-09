# Apicadabri

Apicadabri is a magical set of tools to interact with APIs from a data scientist's perspective to "just get the damn data"™.

It focuses on simplicity and speed while being agnostic about what kind of API you're calling.
If you know how to send a single call to the API you're interested in, you should be good to go to scale up to 100k calls with apicadabri.

## Current status

This is still an early alpha. Some basic examples already work, though (see below).

## Assumptions

For now, apicadabri assumes that you want to solve a task for which the following holds:

* All inputs fit into memory
* ~All results fit into memory~ (you can write directly to a JSONL file)
* The number of requests will not overwhelm the asyncio event loop (which is apparently [hard to achieve](https://stackoverflow.com/questions/55761652/what-is-the-overhead-of-an-asyncio-task) anyway unless you have tens of millions of calls).
* You want to observe and process results as they come in.
* You want your results in the same order as the input with no gaps in between.

### Future relaxing of constraints

* For an extreme numbers of calls (>> 1M), add another layer of batching to avoid creating all asyncio tasks at the same time while also avoiding that one slow call in a batch slows down the whole task.
  * Through the same mechanism, allow loading inputs one batch at a time.

## Examples

### Multiple URLs

```python
import apicadabri
pokemon = ["bulbasaur", "squirtle", "charmander"]
data = apicadabri.bulk_get(
    urls=(f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon),
).to_list()
```

### Multiple payloads

TODO

### Multivariate (zipped)

TODO

### Multivariate (multiply)

TODO

### Multivariate (pipeline)

TODO
