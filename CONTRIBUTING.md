# Contributing to CO2 Leakage Detection System

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## We Develop with Github
We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## We Use [Github Flow](https://guides.github.com/introduction/flow/index.html)
Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Development Process
1. Clone the repository
2. Create a virtual environment and install dependencies:
```bash
make setup
```
3. Run tests:
```bash
make test
```
4. Start the development environment:
```bash
make docker-up
```

## Running Tests
We use pytest for our test suite. To run tests:
```bash
make test
```

## Code Style
We use black and isort for code formatting. Format your code with:
```bash
make format
```

Check code style with:
```bash
make lint
```

## Monitoring Setup
The project includes a comprehensive monitoring stack:
- Prometheus for metrics collection
- Grafana for visualization
- MLflow for experiment tracking

Start the monitoring stack:
```bash
make docker-up
```

## Any contributions you make will be under the MIT Software License
In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using Github's [issue tracker](../../issues)
We use GitHub issues to track public bugs. Report a bug by [opening a new issue](../../issues/new).

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## License
By contributing, you agree that your contributions will be licensed under its MIT License.