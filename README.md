# Mobility Analysis Workflow

MAWpy is a Python package designed for mobility analysis workflows, providing
tools for handling cellular/GPS traces and related data.

This repository contains scripts for processing user cellular/gps trace data
using various techniques to improve data quality and accuracy. The pipeline
involves several steps to handle trace segmentation, incremental clustering,
address oscillation and update stay duration.

## Overview

The pipeline consists of the following processing steps:

1. **Incremental Clustering**: Clusters traces based on a spatial threshold to
   identify potential stay points.
2. **Update Stay Duration**: Updates the duration of identified stays.
3. **Address Oscillation**: Handles oscillations in traces to ensure accurate
   stay detection.
4. **Trace Segmentation Clustering**: Segments traces and clusters them based on
   spatial and duration constraints.

## Prerequisites

- Python >=3.11

## Installation As a Package

To install MAWpy, you can use one of the following methods:

### Using PyPI

The simplest way to install MAWpy is via PyPI using `pip`. This will install the
package along with its dependencies:

```bash
pip install mawpy
```

## Setting Up Your Development Environment

### 1. Create and Activate a Virtual Environment

Create and activate a virtual environment to manage project dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2. Install Project Dependencies

Install the main project dependencies:

```bash
pip install .
```

## 3. Install Development Dependencies

To set up your development environment, you need to install additional
dependencies. These dependencies are necessary for running tests, linting, and
other development tasks.

Run the following command to install all development dependencies:

```bash
pip install ".[dev]"
```

## 4. Install Documentation Dependencies

To build and view the documentation, you need to install the necessary
dependencies. Follow these steps to set up the documentation environment:

Install the documentation dependencies using `pip`. These dependencies are
listed under the `[docs]` optional dependencies in `pyproject.toml`. Run the
following command:

```bash
pip install ".[docs]"
```

## Running Tests

To ensure your changes are properly tested, follow these steps: Execute the test
suite using `pytest`:

```bash
pytest
```
