# TODO

## Definitions

- coroutine
- async
- await
- `async with` = 
- `async for` = allows to loop through an asynchronous iterator. e.g. an iter object that has to be awaited while you
fetch data from it. None of the above would work with an ordinary for, at least not without blocking the event loop. 
This is because for calls __next__ as a blocking function and doesn't await its result. https://stackoverflow.com/questions/56161595/how-to-use-async-for-in-python
- await asyncio.sleep(5)
- task
- future

### Ref

- ref for `async for` and `async with`:
  - https://stackoverflow.com/questions/67092070/why-do-we-need-async-for-and-async-with
  - for:
    - https://stackoverflow.com/questions/56161595/how-to-use-async-for-in-python
  - with:
    - ...