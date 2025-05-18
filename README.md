[![build](https://github.com/CSchoel/apicadabri/actions/workflows/ci.yaml/badge.svg)](https://github.com/CSchoel/apicadabri/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/CSchoel/apicadabri/graph/badge.svg?token=2VMDQFXK3V)](https://codecov.io/gh/CSchoel/apicadabri)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md) 

# Apicadabri

Apicadabri is a magical set of tools to interact with web APIs from a data scientist's perspective to "just get the damn data"™.

Whether you're using raw HTTP calls through `requests` or `aiohttp` or you already have a Python wrapper for your chosen API, apicadabri can probably make your life easier and get you the data faster.
If you know how to send a single call to the API you're interested in, you should be good to go to scale up to 100k calls with a few lines of apicadabri code.

## Current status

This is still in beta phase. Most of the API is stable, but breaking changes might still happen in minor version updates. Most (but not all) of the main features are implemented (see below).

## Features

* 🚀 Get the maximum amount of speed while still playing nice with the API provider.
  * ⚙️ Configurable number of calls active at the same time (using a Semaphore).
  * 🔀 Async execution, so everything stays within one Python process.
* 🐤 You don't have to write `async` or care about task scheduling anywhere.
* 🪜 Process results right as they come in.
* 🐛 Comprehensive error handling and retry mechanisms.
* 📊 Directly get a dataframe from just a single chain of method calls.*
* 🔧 More than just HTTP: Use the abovementioned features for arbitrary (async) tasks.

*: Not yet fully implemented.

## Assumptions

Apicadabri makes the following assumptions about your task:

* 💾 All inputs fit into memory.
* 💾 Outputs may be larger than available runtime memory.
  * ➡️ It must be possible to pipe outputs directly to a file.
* ♾️ Wrapping all inputs into asyncio tasks at the same time will not overwhelm the asyncio event loop.
  * Overwhelming asyncio is apparently [hard to achieve](https://stackoverflow.com/questions/55761652/what-is-the-overhead-of-an-asyncio-task) anyway unless you have tens of millions of calls.
* 👀 Live access to results is important. Fire-and-forget is not good enough.
* 🔢 The order of results must be preserved.
* 🎱 The total number of results must be the same as the number of inputs.
  * If filtering is to happen, it happens after the apicadabri call has finished.

### Future relaxing of constraints

The following changes to the above constraints could happen in the future if it turns out that there are enough use cases:

* For an extreme numbers of calls (>> 1M), one could add another layer of batching to avoid creating all asyncio tasks at the same time while also avoiding that one slow call in a batch slows down the whole task.
* Through the same mechanism, it would be possible to allow loading inputs one batch at a time.

## Examples

### Multiple URLs

```python
import apicadabri
pokemon = ["bulbasaur", "squirtle", "charmander"]
data = apicadabri.bulk_get(
    urls=(f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon),
).json().to_list()
```

### Multiple parameters

```python
import apicadabri
test_inputs = [{"foo": "1" , "bar": "2"}] * 3
data = apicadabri.bulk_get(
  url = "https://httpbin.org/get",
  param_sets = test_inputs
).json().map(lambda r: r["args"]).to_list()
```

### Multiple payloads

```python
import apicadabri
test_inputs = [{"foo": 1 , "bar": 2}] * 3
data = apicadabri.bulk_post(
  url = "https://httpbin.org/post",
  json_sets = test_inputs
).json().map(lambda r: r["data"]).to_list()
```

## Functional API using lazy evaluation

Apicadabri's API (say that fast 10 times) is built with a functional architecture using [map](https://en.wikipedia.org/wiki/Map_(higher-order_function)) and [reduce](https://en.wikipedia.org/wiki/Fold_(higher-order_function)) methods. From the first call to one of the top-level functions to the last step that just gives you the list or DataFrame or whatever you want as output, everything is just a steam of function applications. Consider this example:

```python
import apicadabri
pokemon = ["bulbasaur", "squirtle", "charmander"]
data = (
  apicadabri.bulk_get(
    urls=[f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon],
  )
  .json()
  .map(lambda p: [t["type"]["name"] for t in p["types"]])
  .tqdm(desc="Downloading")
  .tee(lambda res, i, n: print(f"Type of pokemon {i}/{n} is {', '.join(res)}."))
  .to_list()
)
```

All the function and method calls up to `to_list()` actually just build a pipeline.
`bulk_get()` creates a response object that will call the specified API endpoint, but doesn't execute this call yet.
Instead, you first have to define what to do with the data in subsequent method calls.
Each `map`-like method (`json`, `map`, `tqdm`, and `tee`) just wraps the response in another object that contains the code necessary to achieve the goal of that method to the individual results.

Finally, when `to_list()` is called, which is a special case of a `reduce` method, the pipeline is actually activated:
All API calls in `bulk_get()` are executed asynchronously and once the first one returns, it is passed through the entire pipeline, allowing you to inspect it via `tee()` and seet the progress with `tqdm()`.
Once the last result has passed through the whole pipeline, `to_list()` terminates and returns a list of all results.

ℹ️ It is important to note that while `bulk_get()` is asynchronous, the pipeline of `map` and `reduce` calls that follows afterward is executed synchronously again.
This is done for convenience, since apicadabri is built for tasks which have network latency as the main bottleneck.
If computationally expensive postprocessing is required, it is better to just store intermediate results in a DataFrame or similar structure and then process them from there.

## Multivariate calls

If you need to supply mulitple values for more than one parameter of the bulk HTTP call (e.g. supplying both `urls` and `param_sets`), apicadabri supports two separate behaviors chosen via the `mode` parameter.

* `zip` combines the first value of the first parameter with the first value of the second parameter for the first call, and so on (using Python's `zip` function).

    ```python
    import apicadabri
    data = apicadabri.bulk_post(
      url = "https://httpbin.org/post",
      param_sets = [{"foo": "1" , "bar": "2"}] * 2,
      json_sets = [ {"foobar": "bar"} ] * 2,
    ).json().map(lambda r: r["args"]).to_list()
    ```

* `product` builds the carthesian product of all iterable inputs, effectively using all possible combinations of them for the individual calls.

    ```python
    import apicadabri
    data = apicadabri.bulk_post(
      url = "https://httpbin.org/post",
      param_sets = [{"foo": "1" , "bar": "2"}] * 2,
      json_sets = [ {"foobar": "bar"} ] * 2,
      mode = "product",
    ).json().map(lambda r: r["args"]).to_list()
    ```

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

## Controlling retry behavior

By default, all `bulk_*` calls retry an API call up to ten times on any exception that is thrown. If you want to modify this behavior, you can supply an `AsyncRetrier` object in the `retrier` parameter like this:

```python
import apicadabri
pokemon = ["bulbasaur", "squirtle", "charmander"]
apicadabri.bulk_get(
  urls=(f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon),
  retrier=apicadabri.AsyncRetrier(
    max_retries = 3,
    initial_sleep_s = 0.01,
    sleep_multiplier = 2,
    max_sleep_s = 60 * 15,
  ),
).json().to_list()
```

Apart from the maximum number of retries, you can also configure the exponential backoff parameters.
The first retry will wait `initial_sleep_s` seconds before attempting the call again, and after that the sleep time is multiplied by `sleep_multiplier` for every next retry until either the call succeeds, the maximum number of retries is reached.
If during this time the maxmimum sleep time is reached, the multiplier will no longer be applied.

## Tracking task progress

There are two ways of tracking the progress of an apicadabri bulk call:

1. Using the `tqdm()` method of a response object, which will just print a progress bar:

    ```python
    import apicadabri
    data = apicadabri.bulk_get(
      urls=[f"https://pokeapi.co/api/v2/pokemon/{id}" for id in range(1, 5)],
    ).json().tqdm().to_list()
    ```

2. Using the `tee()` method to inject a lambda function that introduces a side-effect and receives the number of items processed as argument.

    ```python
    import apicadabri
    inspect_func = lambda res, i, n: print(f"Halfway done!") if i == int(n / 2) else None
    data = apicadabri.bulk_get(
      urls=[f"https://pokeapi.co/api/v2/pokemon/{id}" for id in range(1, 5)],
    ).json().tee(inspect_func).to_list()

ℹ️ Note that the total number of calls is only known if all user-supplied iterables implement `__len__` or if a size hint was explicitly given with the `size` argument.

## Using apicadabri for arbitrary async tasks

The examples presented so far all use HTTP calls through `aiohttp`.
However, it is entirely possible to use the functionality provided by apicadabry for any arbitrary task involving `async`.

For that, you have to create a subclass of `ApicadabriBulkResponse[A, R]` where `A` is the type of the individual arguments sent to the task and `R` is the type of the results.

The methods you have to overwrite are `call_api`, which implements an invidivual instance of the async call you want to make, and `instances` which returns an interable that supplies the arguments to the individual calls.

The following code shows a trivial example of a task that just returns the length of a string:

```python
class ExampleTask(ApicadabriBulkResponse[str, int]):
    """Test task as example for using Apicadabri without aiotthp."""

    def __init__(self, data: list[str], max_active_calls: int = 10) -> None:
        """Initialize test task."""
        super().__init__(
          max_active_calls=max_active_calls,
          size=len(data)
        )
        self.data = data

    async def call_api(
        self,
        client: ClientSession,
        index: int,
        instance_args: str,
    ) -> tuple[int, int]:
        """Non-aiohttp API call."""
        return (index, len(instance_args))

    def instances(self) -> Iterable[str]:
        """Get instances."""
        return self.data
```