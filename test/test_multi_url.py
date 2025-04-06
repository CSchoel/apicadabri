import apicadabri


def test_multi_url():
    pokemon = ["bulbasaur", "squirtle", "charmander"]
    data = apicadabri.bulk_get(
        urls=(f"https://pokeapi.co/api/v2/pokemon/{p}" for p in pokemon)
    )
    assert data is not None
