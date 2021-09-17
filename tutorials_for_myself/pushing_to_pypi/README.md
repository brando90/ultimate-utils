# Steps to upload to pypi

```angular2html
pip install twine
```

go to project src and do:
```angular2html
python setup.py sdist bdist_wheel
```

create the distribution for pypi:
```angular2html
twine check dist/*
```

## Upload to pytest [optional]

```angular2html
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```
then click the url that appears. e.g.
```angular2html
Uploading distributions to https://test.pypi.org/legacy/
Enter your username: brando90
Enter your password: 
Uploading ultimate_utils-0.1.0-py3-none-any.whl
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 94.2k/94.2k [00:01<00:00, 52.4kB/s]
Uploading ultimate-utils-0.1.0.tar.gz
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 82.5k/82.5k [00:01<00:00, 75.0kB/s]

View at:
https://test.pypi.org/project/ultimate-utils/0.1.0/
```

## Upload to pypi

```angular2html
twine upload dist/*
```
click url that appears to test it worked e.g.
```angular2html
Uploading distributions to https://upload.pypi.org/legacy/
Enter your username: brando90
Enter your password: 
Uploading ultimate_utils-0.1.0-py3-none-any.whl
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 94.2k/94.2k [00:02<00:00, 32.9kB/s]
Uploading ultimate-utils-0.1.0.tar.gz
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 82.5k/82.5k [00:01<00:00, 43.4kB/s]

View at:
https://pypi.org/project/ultimate-utils/0.1.0/
```

# Reference

https://realpython.com/pypi-publish-python-package/#publishing-to-pypi