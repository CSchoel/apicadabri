# Apicadabri

Apicadabri is a magical set of tools to interact with APIs from a data scientist's perspective to "just get the damn data"™.

It focuses on simplicity and speed while being agnostic about what kind of API you're calling.
If you know how to send a single call to the API you're interested in, you should be good to go to scale up to 100k calls with apicadabri.

## Examples

### Multiple URLs

```python
import apicadabri
pokemon = ["bulbasaur", "squirtle", "charmander"]
response = apicadabri.bulk_get(urls=f"https://pokeapi.co/api/v2/{p}" for p in pokemon)
print()
```

### Multiple payloads

TODO

### Multivariate (zipped)

TODO

### Multivariate (multiply)

TODO

### Multivariate (pipeline)

TODO
