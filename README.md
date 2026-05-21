# 🦜🔗 LangChain & Generative AI — Complete Notes
### From Absolute Beginner to Production-Ready Agentic AI

> **How to use these notes:** Read linearly for learning. Each section builds on the previous. Code examples are complete and runnable. Every concept is explained from first principles before going deep.

---

## TABLE OF CONTENTS

1. [What is LangChain & Why It Exists](#1-what-is-langchain--why-it-exists)
2. [Architecture Overview](#2-architecture-overview)
3. [Installation & Environment Setup](#3-installation--environment-setup)
4. [LangChain Core Primitives](#4-langchain-core-primitives)
5. [Models: LLMs & Chat Models](#5-models-llms--chat-models)
6. [Prompts & Prompt Engineering](#6-prompts--prompt-engineering)
7. [Output Parsers](#7-output-parsers)
8. [Chains](#8-chains)
9. [LangChain Expression Language (LCEL)](#9-langchain-expression-language-lcel)
10. [Memory Systems](#10-memory-systems)
11. [Document Loaders](#11-document-loaders)
12. [Text Splitters](#12-text-splitters)
13. [Embeddings](#13-embeddings)
14. [Vector Stores](#14-vector-stores)
15. [Retrievers](#15-retrievers)
16. [RAG Systems](#16-rag-systems)
17. [Tools & Toolkits](#17-tools--toolkits)
18. [Agents](#18-agents)
19. [LangGraph](#19-langgraph)
20. [Multi-Agent Systems](#20-multi-agent-systems)
21. [Callbacks & Observability](#21-callbacks--observability)
22. [LangSmith](#22-langsmith)
23. [Production Patterns](#23-production-patterns)
24. [Evaluation](#24-evaluation)
25. [Common Failure Modes & Debugging](#25-common-failure-modes--debugging)

---

# 1. What is LangChain & Why It Exists

## 1.1 The Problem LangChain Solves

Before LangChain, building applications with Large Language Models (LLMs) meant writing enormous amounts of boilerplate code:

```
❌ Without LangChain:
- Manually format API requests for each model provider
- Write custom retry logic, streaming, error handling
- Build prompt templates from scratch
- Parse raw string outputs manually
- Wire together multiple API calls manually
- No standardized way to add memory/context
- No standard for tool usage or agents
```

LangChain is a **framework** that standardizes and simplifies all of this. Think of it like Express.js for Node, or Django for Python — it doesn't replace the underlying technology, it makes building with it faster and more maintainable.

## 1.2 What is LangChain, Really?

LangChain is a **Python (and JavaScript) framework** for building applications powered by language models. It provides:

| Component | What it does |
|-----------|-------------|
| **Abstractions** | Common interfaces across 100+ LLM providers |
| **Primitives** | Building blocks: prompts, models, parsers, memory |
| **Chains** | Composable pipelines connecting primitives |
| **Agents** | LLMs that decide what actions to take |
| **Integrations** | 100s of data sources, tools, vector databases |
| **Observability** | Tracing, debugging, evaluation via LangSmith |

## 1.3 The Generative AI Ecosystem

To understand LangChain's place, understand the ecosystem:

```
┌─────────────────────────────────────────────────────────┐
│                    YOUR APPLICATION                      │
├─────────────────────────────────────────────────────────┤
│                      LANGCHAIN                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │ Prompts  │  │  Chains  │  │  Agents  │  │  RAG   │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘  │
├─────────────────────────────────────────────────────────┤
│                    LLM PROVIDERS                         │
│  OpenAI  │  Anthropic  │  Google  │  Mistral  │  Ollama │
└─────────────────────────────────────────────────────────┘
```

## 1.4 What is a Large Language Model (LLM)?

An LLM is a neural network trained on massive text datasets to predict the next token (word/subword) in a sequence. Key concepts:

**Token**: The basic unit of text. "Hello world" ≈ 2 tokens. "supercalifragilistic" ≈ 5 tokens. Rule of thumb: 1 token ≈ 0.75 words.

**Context Window**: Maximum tokens the model can process at once. GPT-4 = 128K, Claude = 200K, Gemini = 1M.

**Temperature**: Controls randomness. 0 = deterministic, 1 = creative, >1 = chaotic. Use 0 for facts, 0.7 for creative writing.

**Parameters**: Weight values in the neural network. GPT-4 ≈ 1.76 trillion params. More params ≠ always better.

**Inference**: The process of running the model to generate output. What happens when you call the API.

## 1.5 Types of Models

```
LLMs (Base Models)
  └── Trained to complete text (predict next token)
  └── Example: GPT-3, original LLaMA

Instruction-Tuned Models
  └── Fine-tuned with RLHF to follow instructions
  └── Example: GPT-3.5-turbo, Claude, LLaMA-2-chat

Chat Models
  └── Use conversation format (system/user/assistant)
  └── Example: gpt-4o, claude-3-opus, gemini-pro

Embedding Models
  └── Convert text → fixed-size vectors (numbers)
  └── Example: text-embedding-ada-002, all-MiniLM-L6-v2

Multimodal Models
  └── Process text + images/audio/video
  └── Example: GPT-4V, Claude-3, Gemini Ultra
```

## 1.6 Why Agentic AI?

Traditional software: programmer defines EVERY step.  
Agentic AI: LLM **decides** which steps to take based on the goal.

```
Traditional: User asks "What's the weather?" 
→ Code calls weather API
→ Returns result

Agentic: User asks "Plan my week based on weather"
→ LLM decides: 1) Check calendar  2) Check weather  3) Cross-reference
→ LLM calls tools in right order
→ LLM synthesizes final plan
```

This is the paradigm shift LangChain is built for.

---

# 2. Architecture Overview

## 2.1 LangChain Package Structure

LangChain is split into multiple packages (as of v0.3):

```
langchain                  ← Main package, chains, agents
langchain-core             ← Base abstractions, interfaces
langchain-community        ← 3rd-party integrations
langchain-openai           ← OpenAI-specific
langchain-anthropic        ← Anthropic-specific
langchain-google-genai     ← Google-specific
langgraph                  ← Agent orchestration graphs
langsmith                  ← Observability & evaluation
```

**Why split?** Dependency management. You don't need to install all of Hugging Face just to use OpenAI.

## 2.2 The Runnable Protocol

The **most important concept** in LangChain is the `Runnable` interface. Almost everything is a Runnable:

```python
# Every Runnable has these methods:
runnable.invoke(input)          # Single call, synchronous
runnable.batch([input1, input2]) # Multiple inputs, parallel
runnable.stream(input)          # Streaming output
await runnable.ainvoke(input)   # Async single
await runnable.abatch([...])    # Async batch
runnable.astream(input)         # Async stream
```

This uniformity means you can swap any component for another with the same interface.

## 2.3 The Data Flow

```
Input (dict/string)
    ↓
PromptTemplate      ← Formats the input into a prompt
    ↓
ChatModel           ← Sends to LLM, gets AIMessage
    ↓
OutputParser        ← Converts AIMessage to usable format
    ↓
Output (string/dict/object)
```

## 2.4 Component Hierarchy

```
LangChain Components
├── Input/Output
│   ├── Messages (HumanMessage, AIMessage, SystemMessage)
│   ├── Prompt Templates
│   └── Output Parsers
├── Models
│   ├── LLMs (text in → text out)
│   └── Chat Models (messages in → message out)
├── Retrieval
│   ├── Document Loaders
│   ├── Text Splitters
│   ├── Embeddings
│   ├── Vector Stores
│   └── Retrievers
├── Chains (pre-built pipelines)
├── Agents
│   ├── Agent (the LLM-based decision maker)
│   ├── Tools (functions the agent can call)
│   └── AgentExecutor (runs the agent loop)
└── Memory (conversation history)
```

---

# 3. Installation & Environment Setup

## 3.1 Prerequisites

You need Python 3.9+ and pip. Check:
```bash
python --version    # Should be 3.9+
pip --version
```

## 3.2 Installation

```bash
# Core packages
pip install langchain langchain-core langchain-community

# Provider packages (install what you need)
pip install langchain-openai        # OpenAI + Azure OpenAI
pip install langchain-anthropic     # Anthropic Claude
pip install langchain-google-genai  # Google Gemini
pip install langchain-ollama        # Local models via Ollama

# Observability
pip install langsmith

# Agent orchestration
pip install langgraph

# Common utilities
pip install python-dotenv           # Load .env files
pip install tiktoken                # Token counting for OpenAI
```

## 3.3 API Keys & Environment Variables

**NEVER hardcode API keys in your code.** Use environment variables:

```bash
# Create a .env file (add this to .gitignore!)
touch .env
echo "OPENAI_API_KEY=sk-..." >> .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
echo "LANGCHAIN_API_KEY=ls__..." >> .env
echo "LANGCHAIN_TRACING_V2=true" >> .env
echo "LANGCHAIN_PROJECT=my-project" >> .env
```

```python
# In your Python file, load the .env file first
from dotenv import load_dotenv
import os

load_dotenv()  # Loads all variables from .env

# Access them
api_key = os.getenv("OPENAI_API_KEY")
```

## 3.4 Your First LangChain Program

```python
# hello_langchain.py
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

# Initialize the model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Send a message
response = llm.invoke([HumanMessage(content="What is 2 + 2?")])

print(response.content)  # "2 + 2 equals 4."
print(type(response))    # <class 'langchain_core.messages.ai.AIMessage'>
```

## 3.5 Using Ollama (Free, Local Models)

Ollama lets you run models locally, no API key needed:

```bash
# Install Ollama from https://ollama.ai
# Pull a model
ollama pull llama3.2
ollama pull mistral
ollama pull nomic-embed-text  # For embeddings
```

```python
from langchain_ollama import ChatOllama, OllamaEmbeddings

llm = ChatOllama(model="llama3.2", temperature=0)
response = llm.invoke("Hello!")
print(response.content)
```

## 3.6 Project Structure (Best Practice)

```
my_langchain_project/
├── .env                    # API keys (NEVER commit this)
├── .gitignore              # Include .env here
├── requirements.txt        # pip freeze > requirements.txt
├── main.py                 # Entry point
├── chains/                 # Your custom chains
│   ├── __init__.py
│   └── qa_chain.py
├── agents/                 # Your agents
│   ├── __init__.py
│   └── research_agent.py
├── prompts/                # Prompt templates
│   └── templates.py
└── utils/                  # Helpers
    └── helpers.py
```

---

# 4. LangChain Core Primitives

## 4.1 Messages

Messages are the fundamental unit of communication with chat models. There are 5 types:

```python
from langchain_core.messages import (
    SystemMessage,    # Sets model behavior/persona
    HumanMessage,     # User input
    AIMessage,        # Model response
    ToolMessage,      # Result from a tool call
    FunctionMessage,  # Legacy, prefer ToolMessage
)

# SystemMessage: Instructions that set the context
system = SystemMessage(content="You are a helpful Python tutor.")

# HumanMessage: What the user says
human = HumanMessage(content="Explain list comprehensions")

# AIMessage: What the model responds
# (You usually receive these, not create them)
ai = AIMessage(content="List comprehensions are...")

# Constructing a conversation manually
messages = [
    SystemMessage(content="You are a pirate. Speak like one."),
    HumanMessage(content="What's your name?"),
    AIMessage(content="Arrr, they call me Captain Bytes!"),
    HumanMessage(content="Where do you sail?"),
]
```

**AIMessage anatomy:**
```python
response = llm.invoke(messages)
print(response.content)          # "I sail the digital seas!"
print(response.response_metadata) # tokens used, model name, etc.
print(response.id)               # Unique message ID
print(response.tool_calls)       # If model called a tool
```

## 4.2 Documents

A `Document` represents a piece of text with metadata:

```python
from langchain_core.documents import Document

doc = Document(
    page_content="LangChain is a framework for LLM applications.",
    metadata={
        "source": "langchain_docs.pdf",
        "page": 1,
        "author": "Harrison Chase",
        "date": "2024-01-01"
    }
)

print(doc.page_content)  # The text
print(doc.metadata)      # The metadata dict
```

Documents are what flow through RAG pipelines — they're created by loaders, split by splitters, embedded, stored in vector stores, and retrieved for context.

## 4.3 The Runnable Interface (Deep Dive)

```python
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

# RunnableLambda: Wrap any function as a Runnable
def add_exclamation(text: str) -> str:
    return text + "!"

runnable = RunnableLambda(add_exclamation)
result = runnable.invoke("Hello")  # "Hello!"

# RunnablePassthrough: Pass input through unchanged
passthrough = RunnablePassthrough()
result = passthrough.invoke({"key": "value"})  # {"key": "value"}

# RunnableParallel: Run multiple runnables at the same time
parallel = RunnableParallel(
    doubled=RunnableLambda(lambda x: x * 2),
    tripled=RunnableLambda(lambda x: x * 3),
)
result = parallel.invoke(5)  # {"doubled": 10, "tripled": 15}
```

## 4.4 The Pipe Operator (|)

The `|` operator chains Runnables together. The output of the left becomes the input of the right:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
model = ChatOpenAI()
parser = StrOutputParser()

# Build a chain with the pipe operator
chain = prompt | model | parser

# Run it
result = chain.invoke({"topic": "programming"})
print(result)  # "Why do programmers prefer dark mode? Because light attracts bugs!"
```

This is the **LCEL (LangChain Expression Language)** pattern — the modern way to build chains.

---

# 5. Models: LLMs & Chat Models

## 5.1 LLMs vs Chat Models

```
LLMs (Text Completion)           Chat Models (Conversation)
─────────────────────           ──────────────────────────
Input:  "The sky is"            Input: [HumanMessage("What color is sky?")]
Output: " blue."                Output: AIMessage("The sky is blue.")

Used for: simple completion     Used for: conversation, instruction following
Example: text-davinci-003       Example: gpt-4o, claude-3, gemini-pro
Status: ⚠️ Being deprecated    Status: ✅ Recommended
```

**In practice, always use Chat Models.** They're more capable, support system prompts, and are the standard going forward.

## 5.2 OpenAI Models

```python
from langchain_openai import ChatOpenAI, OpenAI

# Chat model (recommended)
llm = ChatOpenAI(
    model="gpt-4o",              # Model name
    temperature=0.7,              # Creativity (0-2)
    max_tokens=1000,              # Max response length
    timeout=30,                   # Request timeout seconds
    max_retries=2,                # Auto-retry on failure
    api_key="sk-...",            # Or use OPENAI_API_KEY env var
)

# Available models (as of 2024):
# gpt-4o          - Most capable, vision support
# gpt-4o-mini     - Faster, cheaper, still great
# gpt-3.5-turbo   - Fast, cheap, less capable
# o1-preview      - Reasoning model (slow, expensive)
# o1-mini         - Smaller reasoning model

# Invoke (single call)
response = llm.invoke("What is machine learning?")
print(response.content)

# Invoke with messages
from langchain_core.messages import SystemMessage, HumanMessage
response = llm.invoke([
    SystemMessage(content="You are a machine learning expert."),
    HumanMessage(content="Explain neural networks simply.")
])

# Streaming
for chunk in llm.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

## 5.3 Anthropic Claude Models

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0,
    max_tokens=2048,
    # ANTHROPIC_API_KEY env var is used automatically
)

# Available models:
# claude-3-5-sonnet-20241022  - Best balance of speed/intelligence
# claude-3-5-haiku-20241022   - Fastest, cheapest
# claude-3-opus-20240229      - Most intelligent

response = llm.invoke("Explain quantum computing in simple terms")
print(response.content)
```

## 5.4 Google Gemini Models

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    # GOOGLE_API_KEY env var
)

# Available models:
# gemini-1.5-pro    - 1M context, multimodal
# gemini-1.5-flash  - Fast, efficient
# gemini-pro        - Original

response = llm.invoke("What is 15% of 340?")
```

## 5.5 Local Models with Ollama

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2",      # Model you pulled with ollama pull
    temperature=0,
    num_ctx=4096,          # Context window size
)

response = llm.invoke("Hello! Who are you?")
```

## 5.6 Model Configuration Patterns

```python
# Bind parameters to a model (returns new model with those params)
precise_llm = llm.bind(temperature=0, max_tokens=100)
creative_llm = llm.bind(temperature=1.2, max_tokens=500)

# Bind a stop sequence (stop generating at this string)
llm_with_stop = llm.bind(stop=["END", "STOP"])

# Structured output (model returns JSON matching a schema)
from pydantic import BaseModel

class MovieReview(BaseModel):
    title: str
    rating: float
    summary: str

structured_llm = llm.with_structured_output(MovieReview)
review = structured_llm.invoke("Review The Matrix")
print(review.title)   # "The Matrix"
print(review.rating)  # 9.5
print(review.summary) # "A groundbreaking sci-fi..."
```

## 5.7 Token Counting & Cost Management

```python
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import UsageMetadataCallbackHandler

# Track usage
callback = UsageMetadataCallbackHandler()
llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[callback])

response = llm.invoke("What is Python?")
print(callback.usage_metadata)
# {'input_tokens': 15, 'output_tokens': 200, 'total_tokens': 215}

# Check token count before sending (OpenAI models)
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")
tokens = enc.encode("Hello, how are you?")
print(f"Token count: {len(tokens)}")  # 6

# Cost estimation (approximate):
# GPT-4o:      $2.50/1M input tokens,  $10/1M output tokens
# GPT-4o-mini: $0.15/1M input tokens,  $0.60/1M output tokens
# Claude Haiku: $0.25/1M input tokens, $1.25/1M output tokens
```

## 5.8 Fallbacks

```python
# If primary model fails, fall back to another
fast_llm = ChatOpenAI(model="gpt-4o-mini")
capable_llm = ChatOpenAI(model="gpt-4o")

# If fast_llm fails, use capable_llm
llm_with_fallback = fast_llm.with_fallbacks([capable_llm])

response = llm_with_fallback.invoke("Hello")
```

---

# 6. Prompts & Prompt Engineering

## 6.1 Why Prompt Templates?

Hard-coded strings are brittle. Templates are reusable, maintainable, and can be:
- Versioned and stored
- Shared across chains
- Dynamically composed

## 6.2 PromptTemplate (Simple String)

```python
from langchain_core.prompts import PromptTemplate

# Basic template
template = PromptTemplate(
    input_variables=["topic", "style"],
    template="Explain {topic} in the style of {style}."
)

# Format it
prompt = template.format(topic="blockchain", style="a 5-year-old")
print(prompt)
# "Explain blockchain in the style of a 5-year-old."

# Shortcut
template = PromptTemplate.from_template("Explain {topic} in the style of {style}.")
```

## 6.3 ChatPromptTemplate (Modern Standard)

```python
from langchain_core.prompts import ChatPromptTemplate

# From messages
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert {field} teacher."),
    ("human", "Teach me about {topic}. Use {num_examples} examples."),
])

# Invoke it
messages = prompt.invoke({
    "field": "chemistry",
    "topic": "covalent bonds",
    "num_examples": 3
})
# Returns a list of formatted messages

# From template string
prompt = ChatPromptTemplate.from_template("What is {thing}?")
```

## 6.4 Multi-Turn Conversation Templates

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# MessagesPlaceholder inserts a list of messages at that position
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),  # ← Insert history here
    ("human", "{input}"),
])

# Use with history
from langchain_core.messages import HumanMessage, AIMessage

history = [
    HumanMessage(content="My name is Alice."),
    AIMessage(content="Nice to meet you, Alice!"),
]

messages = prompt.invoke({
    "chat_history": history,
    "input": "What's my name?"
})
# System message + history messages + new human message
```

## 6.5 Few-Shot Prompting

Provide examples to guide the model's behavior:

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate

# Define your examples
examples = [
    {
        "input": "happy",
        "output": "sad"
    },
    {
        "input": "tall",
        "output": "short"
    },
    {
        "input": "fast",
        "output": "slow"
    },
]

# Create the example prompt format
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "What is the opposite of {input}?"),
    ("ai", "{output}"),
])

# Build few-shot prompt
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# Combine with the actual task
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You answer antonym questions."),
    few_shot_prompt,                        # Insert examples here
    ("human", "What is the opposite of {input}?"),
])

chain = final_prompt | ChatOpenAI() | StrOutputParser()
result = chain.invoke({"input": "loud"})
print(result)  # "quiet"
```

## 6.6 Prompt Engineering Principles

### Principle 1: Be Specific and Detailed
```python
# ❌ Vague
"Summarize this"

# ✅ Specific
"Summarize the following article in exactly 3 bullet points. 
Each bullet should be one sentence. Focus on the key findings, 
not background information. Article: {article}"
```

### Principle 2: Use Role/Persona
```python
# Giving the model a role dramatically improves output quality
system = """You are a senior software engineer with 10 years of Python experience.
You write clean, idiomatic, well-commented code. You always consider edge cases
and performance implications. When reviewing code, you are constructive but honest."""
```

### Principle 3: Output Format Instructions
```python
# Tell the model exactly how to format its response
template = """Analyze the following code for bugs.

Code:
{code}

Respond in this EXACT format:
BUGS FOUND: <number>
SEVERITY: <low/medium/high/critical>
BUGS:
1. Line <N>: <description>
2. Line <N>: <description>
RECOMMENDATION: <one sentence fix suggestion>"""
```

### Principle 4: Chain of Thought
```python
# Ask the model to reason step by step
template = """Solve this math problem step by step, showing all work.

Problem: {problem}

Think through it carefully:
Step 1: Identify what's being asked
Step 2: Identify the relevant information
Step 3: Apply the formula/approach
Step 4: Calculate
Step 5: Check your answer
Final Answer: <answer>"""
```

### Principle 5: Use Delimiters
```python
# Use delimiters to clearly separate parts of the prompt
template = """Translate the text delimited by triple backticks into {language}.

```{text}```

Translation:"""
```

## 6.7 Dynamic Few-Shot Selection

Select relevant examples dynamically from a large pool:

```python
from langchain_core.prompts import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Large pool of examples
all_examples = [
    {"input": "python list", "output": "How to use Python lists..."},
    {"input": "sql join", "output": "SQL JOIN combines rows..."},
    {"input": "css flexbox", "output": "CSS Flexbox is a layout..."},
    # ... hundreds more
]

# Select most similar examples to the user's question
selector = SemanticSimilarityExampleSelector.from_examples(
    all_examples,
    OpenAIEmbeddings(),
    FAISS,
    k=3  # Select top 3 most similar examples
)

# The selector picks examples most similar to the input
selected = selector.select_examples({"input": "python dictionary"})
# Returns examples about Python data structures
```

---

# 7. Output Parsers

## 7.1 Why Output Parsers?

LLMs return raw text. Your application needs structured data. Output parsers bridge this gap.

```
LLM Output (raw text)  →  Output Parser  →  Usable Python object
"The answer is 42."    →  StrOutputParser →  "The answer is 42."
'{"score": 8.5}'       →  JsonOutputParser → {"score": 8.5}
"Title: ..."           →  PydanticParser   → MovieReview(title=...)
```

## 7.2 StrOutputParser

The simplest — extracts the string content:

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

# Converts AIMessage → string
ai_message = AIMessage(content="Hello there!")
result = parser.invoke(ai_message)
print(result)        # "Hello there!"
print(type(result))  # <class 'str'>

# In a chain
chain = prompt | model | StrOutputParser()
```

## 7.3 JsonOutputParser

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

# Use in a chain with format instructions
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""Return a JSON object with keys 'name', 'age', 'city' 
    for this person description: {description}
    
    Return ONLY valid JSON, no other text.""",
    input_variables=["description"]
)

chain = prompt | ChatOpenAI() | JsonOutputParser()
result = chain.invoke({"description": "Alice, 30 years old, lives in NYC"})
print(result)  # {'name': 'Alice', 'age': 30, 'city': 'New York City'}
print(type(result))  # <class 'dict'>
```

## 7.4 PydanticOutputParser

The most powerful — validates against a schema:

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# Define your schema
class Recipe(BaseModel):
    name: str = Field(description="Name of the recipe")
    ingredients: List[str] = Field(description="List of ingredients")
    cook_time_minutes: int = Field(description="Cooking time in minutes")
    difficulty: str = Field(description="easy/medium/hard")
    
# Create parser
parser = PydanticOutputParser(pydantic_object=Recipe)

# Get format instructions to include in prompt
format_instructions = parser.get_format_instructions()
print(format_instructions)
# The output should be formatted as a JSON instance that conforms to 
# the JSON schema below...

prompt = PromptTemplate(
    template="Generate a recipe for {dish}.\n{format_instructions}",
    input_variables=["dish"],
    partial_variables={"format_instructions": format_instructions}
)

chain = prompt | ChatOpenAI() | parser
recipe = chain.invoke({"dish": "pasta carbonara"})

print(recipe.name)             # "Spaghetti Carbonara"
print(recipe.ingredients)      # ["spaghetti", "eggs", "pancetta", ...]
print(recipe.cook_time_minutes) # 25
print(recipe.difficulty)       # "medium"
print(type(recipe))           # <class '__main__.Recipe'>
```

## 7.5 CommaSeparatedListOutputParser

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

prompt = PromptTemplate(
    template="List 5 {thing}s. Return as comma-separated values.",
    input_variables=["thing"],
)

chain = prompt | ChatOpenAI() | parser
result = chain.invoke({"thing": "programming language"})
print(result)  # ['Python', 'JavaScript', 'Java', 'C++', 'Rust']
```

## 7.6 OutputFixingParser

Automatically retries if parsing fails:

```python
from langchain.output_parsers import OutputFixingParser

# Wrap any parser with error-fixing
fixing_parser = OutputFixingParser.from_llm(
    parser=PydanticOutputParser(pydantic_object=Recipe),
    llm=ChatOpenAI()
)

# If the LLM returns malformed JSON, the fixing parser
# sends it back to the LLM with "Fix this JSON: ..."
```

## 7.7 Structured Output (Modern Approach)

The cleanest approach — use `.with_structured_output()`:

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    occupation: str

# Model handles formatting internally (no prompt instructions needed)
structured_llm = ChatOpenAI(model="gpt-4o").with_structured_output(Person)

person = structured_llm.invoke("Tell me about Marie Curie")
print(person.name)       # "Marie Curie"
print(person.age)        # 66 (at death)
print(person.occupation) # "Physicist and Chemist"

# Also works with plain dicts
schema = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "reason": {"type": "string"}
    }
}
structured_llm = ChatOpenAI().with_structured_output(schema)
```

---

# 8. Chains

## 8.1 What is a Chain?

A Chain is a **sequence of calls** — to models, parsers, tools, databases, or any other component — that transforms an input into an output.

```
Input → [Step 1] → [Step 2] → [Step 3] → Output
         ↑ Call    ↑ Transform  ↑ Parse
         LLM       text         to object
```

## 8.2 LLMChain (Legacy, Still Important to Know)

```python
# ⚠️ Legacy API but you'll see this in older code
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template("What is the capital of {country}?")

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.invoke({"country": "France"})
print(result["text"])  # "The capital of France is Paris."
```

## 8.3 Sequential Chain

Run chains one after another, passing output to next:

```python
# Modern LCEL approach
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Chain 1: Generate a topic
generate_topic = (
    ChatPromptTemplate.from_template("Generate a random interesting topic about {field}.")
    | ChatOpenAI()
    | StrOutputParser()
)

# Chain 2: Write about the topic
write_about = (
    ChatPromptTemplate.from_template("Write 3 fascinating facts about: {topic}")
    | ChatOpenAI()
    | StrOutputParser()
)

# Combine: output of chain 1 becomes input for chain 2
combined = generate_topic | (lambda topic: {"topic": topic}) | write_about

result = combined.invoke({"field": "astronomy"})
print(result)
```

## 8.4 ConversationChain (Legacy Memory Chain)

```python
# Legacy but important for understanding memory
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

memory = ConversationBufferMemory()
chain = ConversationChain(
    llm=ChatOpenAI(),
    memory=memory,
    verbose=True  # Prints what's happening
)

response1 = chain.predict(input="My name is Bob.")
response2 = chain.predict(input="What's my name?")
print(response2)  # "Your name is Bob!"
```

## 8.5 RetrievalQA Chain (Legacy RAG)

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Assume vectorstore is already created (covered in RAG section)
vectorstore = FAISS.load_local("my_vectorstore", OpenAIEmbeddings())

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    chain_type="stuff",       # "stuff", "map_reduce", "refine"
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
)

result = qa_chain.invoke({"query": "What is LangChain?"})
print(result["result"])           # The answer
print(result["source_documents"]) # Documents used
```

## 8.6 Summarization Chains

Different strategies for summarizing long documents:

```python
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

llm = ChatOpenAI()

# Strategy 1: "stuff" — put everything in one prompt (good for short docs)
chain = load_summarize_chain(llm, chain_type="stuff")

# Strategy 2: "map_reduce" — summarize each chunk, then summarize summaries
# Good for long documents that exceed context window
chain = load_summarize_chain(llm, chain_type="map_reduce")

# Strategy 3: "refine" — iteratively refine summary with each chunk
# Produces best quality but slowest
chain = load_summarize_chain(llm, chain_type="refine")

# Use it
docs = [Document(page_content="Very long text here...")]
summary = chain.invoke({"input_documents": docs})
print(summary["output_text"])
```

## 8.7 Router Chain

Direct input to different chains based on content:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# Specialized chains
math_chain = (
    ChatPromptTemplate.from_template("Solve this math problem: {input}")
    | ChatOpenAI()
    | StrOutputParser()
)

code_chain = (
    ChatPromptTemplate.from_template("Write Python code for: {input}")
    | ChatOpenAI()
    | StrOutputParser()
)

general_chain = (
    ChatPromptTemplate.from_template("Answer this question: {input}")
    | ChatOpenAI()
    | StrOutputParser()
)

# Router — decides which chain to use
def route(info):
    topic = info["topic"]
    if "math" in topic.lower() or "calculate" in topic.lower():
        return math_chain
    elif "code" in topic.lower() or "program" in topic.lower():
        return code_chain
    else:
        return general_chain

# Classification chain
classifier_prompt = ChatPromptTemplate.from_template(
    "Classify this query into: math, code, or general. Query: {input}\nAnswer with one word:"
)
classifier = classifier_prompt | ChatOpenAI() | StrOutputParser()

# Full routing chain
from langchain_core.runnables import RunnablePassthrough

full_chain = (
    RunnablePassthrough.assign(topic=classifier)  # Add 'topic' key with classification
    | RunnableLambda(route)                        # Route to appropriate chain
)

result = full_chain.invoke({"input": "Calculate 15% of 350"})
```
