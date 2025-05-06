[![build](https://github.com/CSchoel/apicadabri/actions/workflows/ci.yaml/badge.svg)](https://github.com/CSchoel/apicadabri/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/CSchoel/apicadabri/graph/badge.svg?token=2VMDQFXK3V)](https://codecov.io/gh/CSchoel/apicadabri)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md) 

# Apicadabri

Apicadabri is a magical set of tools to interact with web APIs from a data scientist's perspective to "just get the damn data"™.

It focuses on simplicity and speed while being agnostic about what kind of API you're calling.
If you know how to send a single call to the API you're interested in, you should be good to go to scale up to 100k calls with apicadabri.

## Current status

This is still an early alpha. Some basic examples already work, though (see below).

## Features

* 🚀 Get the maximum amount of speed while still playing nice with the API provider.
  * ⚙️ Configurable number of calls active at the same time (using a Semaphore).
  * 🔀 Async execution, so everything stays within one Python process.
* 🐤 You don't have to write `async` or care about task scheduling anywhere.
* 🪜 Process results right as they come in.
* 🐛 Comprehensive error handling and retry mechanisms.*
* 📊 Directly get a dataframe from just a single chain of method calls.*
* 🔧 More than just HTTP: Use the abovementioned features for arbitrary (async) tasks.

*: Not yet fully implemented.

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
).json().to_list()
```

### Multiple payloads

TODO

### Multivariate (zipped)

TODO

### Multivariate (multiply)

TODO

### Multivariate (pipeline)

TODO

## Error Handling

API calls can always fail and you don't want your script with 100k API calls to crash on call number 10k because you forgot to handle a `None` somewhere.
At the same time, though, you might not even care about errors and just want to set up a test scenario quick and dirty.
Apicadabri adapts to both scenarios, by providing you three options for error handling, managed by the `on_error` parameter:

* `raise`: The exception is not caught at all, instead it is just raised as normal and the bulk call will fail.
* `return`: The exception is caught and encapsulated in an `ApicadabriErrorResponse` object, that also contains the input that triggered the exception.
* A lambda function: The exception is caught and the provided error handling function is called with the triggering input and the error message and type.
    The error handling function must return a result of the same type as would be expected by a successful call.
    This can, for example, be used to return an "empty" result that does not lead to exceptions in further processing.

    ℹ️ If you need to return a _different_ type of object in case of an error, you can instead use `map` with `on_error="return"` and then do another `map` that transforms the error response into the type you want.

The `on_error` parameter is available for multiple central methods of return objects, most notably `map` and `reduce`.