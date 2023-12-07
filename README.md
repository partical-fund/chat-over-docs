# 💬 📚 Chat Over Docs

This prototype allows users to chat over their documentns.

## Table of Contents
- [Quick Start](#quick-start)
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Usage](#usage)

## Quick Start

```docker build -t chat_over_docs .```

```docker run -it --name chat_over_docs chat_over_docs /bin/bash```

```vi chat_over_docs.py```

Paste in your OpenAI secret key

Copy your directory of documents into the container.

```docker cp /path/to/local/directory chat_over_docs:/app```

```python chat_over_docs.py```

Enter the path of 

## Project Overview

Provide a more detailed description of your project. Explain its purpose, features, and any other relevant information.

## Prerequisites

List any software, libraries, or other dependencies that users need to have installed before they can use your project.

## Getting Started

Guide users through getting your project up and running on their local machine.