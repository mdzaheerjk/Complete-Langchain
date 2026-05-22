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













---

# 9. LangChain Expression Language (LCEL)

## 9.1 What is LCEL?

LCEL is the **modern, recommended** way to build chains in LangChain. It uses Python's `|` operator to compose Runnables. Every component in LCEL:

- Is a `Runnable`
- Supports `.invoke()`, `.stream()`, `.batch()`, async variants
- Can be composed with `|`
- Automatically handles type conversion between components

```python
# This is LCEL:
chain = prompt | model | parser

# Equivalent to this (verbose version):
class MyChain:
    def invoke(self, input):
        formatted = prompt.invoke(input)
        response = model.invoke(formatted)
        parsed = parser.invoke(response)
        return parsed
```

## 9.2 Core LCEL Patterns

### Pattern 1: Basic Pipeline
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("Tell me about {topic}")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

result = chain.invoke({"topic": "quantum physics"})
```

### Pattern 2: RunnablePassthrough — Pass Input Unchanged
```python
from langchain_core.runnables import RunnablePassthrough

# Keep original input AND add new keys
chain = RunnablePassthrough.assign(
    upper_topic=lambda x: x["topic"].upper()
)

result = chain.invoke({"topic": "python"})
print(result)  # {"topic": "python", "upper_topic": "PYTHON"}
```

### Pattern 3: RunnableParallel — Execute Simultaneously
```python
from langchain_core.runnables import RunnableParallel

# Run two chains at the same time
parallel_chain = RunnableParallel(
    summary=ChatPromptTemplate.from_template("Summarize: {text}") | ChatOpenAI() | StrOutputParser(),
    keywords=ChatPromptTemplate.from_template("Extract keywords from: {text}") | ChatOpenAI() | StrOutputParser(),
)

result = parallel_chain.invoke({"text": "LangChain is a framework..."})
print(result["summary"])   # The summary
print(result["keywords"])  # The keywords
# Note: Both run simultaneously → faster than sequential
```

### Pattern 4: RunnableLambda — Inline Functions
```python
from langchain_core.runnables import RunnableLambda

# Wrap a function as a Runnable
def count_words(text: str) -> dict:
    return {"text": text, "word_count": len(text.split())}

chain = (
    ChatPromptTemplate.from_template("Describe {topic} in one paragraph")
    | ChatOpenAI()
    | StrOutputParser()
    | RunnableLambda(count_words)  # Count words in the response
)

result = chain.invoke({"topic": "AI"})
print(result["word_count"])  # Number of words in response
```

### Pattern 5: ItemGetter — Extract from Dicts
```python
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

# Extract specific keys from a dict
chain = (
    RunnablePassthrough.assign(
        answer=itemgetter("question") | ChatOpenAI() | StrOutputParser()
    )
)
```

## 9.3 Streaming with LCEL

```python
chain = (
    ChatPromptTemplate.from_template("Write a haiku about {topic}")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# Stream token by token
for token in chain.stream({"topic": "coding"}):
    print(token, end="", flush=True)
print()  # New line at end

# Async streaming
async def stream_example():
    async for token in chain.astream({"topic": "coding"}):
        print(token, end="", flush=True)
```

## 9.4 Batch Processing

```python
# Process multiple inputs efficiently (parallelized)
chain = (
    ChatPromptTemplate.from_template("What is the capital of {country}?")
    | ChatOpenAI()
    | StrOutputParser()
)

countries = [
    {"country": "France"},
    {"country": "Japan"},
    {"country": "Brazil"},
    {"country": "Egypt"},
]

# Processes all at once (respects API rate limits automatically)
results = chain.batch(countries, config={"max_concurrency": 3})
print(results)  # ["Paris", "Tokyo", "Brasília", "Cairo"]
```

## 9.5 Error Handling in LCEL

```python
from langchain_core.runnables import RunnableLambda

def risky_function(input):
    if not input.get("name"):
        raise ValueError("Name is required!")
    return f"Hello {input['name']}"

# With fallback
safe_chain = (
    RunnableLambda(risky_function)
    .with_fallbacks([
        RunnableLambda(lambda x: "Hello stranger!")
    ])
)

result = safe_chain.invoke({})  # "Hello stranger!" (not an error)
result = safe_chain.invoke({"name": "Alice"})  # "Hello Alice"

# With retry
retrying_chain = (
    ChatOpenAI()
    .with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True
    )
)
```

## 9.6 Configuring Chains at Runtime

```python
from langchain_core.runnables import ConfigurableField

# Make a parameter configurable at runtime
llm = ChatOpenAI(temperature=0).configurable_fields(
    temperature=ConfigurableField(
        id="temperature",
        name="LLM Temperature",
        description="Temperature of the language model"
    ),
    model=ConfigurableField(
        id="model",
        name="Model Name",
    )
)

chain = prompt | llm | StrOutputParser()

# Configure at runtime
creative_result = chain.with_config(
    configurable={"temperature": 1.0, "model": "gpt-4o"}
).invoke({"topic": "poetry"})

precise_result = chain.with_config(
    configurable={"temperature": 0.0, "model": "gpt-4o-mini"}
).invoke({"topic": "math"})
```

## 9.7 The Full LCEL Reference

```python
# All Runnable methods:
chain.invoke(input)                    # Sync single
chain.batch([input1, input2])          # Sync batch
chain.stream(input)                    # Sync stream
await chain.ainvoke(input)             # Async single  
await chain.abatch([input1, input2])   # Async batch
chain.astream(input)                   # Async stream generator

# Modifiers:
chain.with_config(config)              # Add runtime config
chain.with_retry(...)                  # Add retry logic
chain.with_fallbacks([...])            # Add fallbacks
chain.with_listeners(...)              # Add event listeners
chain.bind(**kwargs)                   # Bind parameters

# Introspection:
chain.input_schema                     # Pydantic schema of expected input
chain.output_schema                    # Pydantic schema of output
chain.get_graph()                      # Get execution graph
```

---

# 10. Memory Systems

## 10.1 The Memory Problem

LLMs are **stateless** by default. Each API call is completely independent:

```
User: "My name is Alice."  → LLM: "Hello Alice!"
User: "What's my name?"    → LLM: "I don't know your name." ❌
```

Memory systems solve this by storing and retrieving conversation history.

## 10.2 Types of Memory

```
Memory Types:
├── Buffer Memory        — Keep all messages
├── Buffer Window        — Keep last N messages  
├── Summary Memory       — LLM-summarizes old messages
├── Summary Buffer       — Hybrid: recent messages + summary of old
├── Entity Memory        — Extract and store key entities
├── Knowledge Graph      — Store as graph relationships
└── Vector Store Memory  — Semantic similarity search
```

## 10.3 Modern Memory with LCEL (Recommended Approach)

```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Simple in-memory store
store = {}  # session_id → ChatMessageHistory

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Build base chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),  # History goes here
    ("human", "{input}"),
])

chain = prompt | ChatOpenAI() | StrOutputParser()

# Wrap with memory
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Use it — same session_id = same conversation
config = {"configurable": {"session_id": "user_123"}}

r1 = chain_with_memory.invoke({"input": "My name is Bob."}, config=config)
print(r1)  # "Nice to meet you, Bob!"

r2 = chain_with_memory.invoke({"input": "What's my name?"}, config=config)
print(r2)  # "Your name is Bob!"

# Different session = different conversation
other_config = {"configurable": {"session_id": "user_456"}}
r3 = chain_with_memory.invoke({"input": "What's my name?"}, config=other_config)
print(r3)  # "I don't know your name yet. What is it?"
```

## 10.4 Persistent Memory with Redis

```python
from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_redis_history(session_id: str):
    return RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379",
        ttl=3600  # Expire after 1 hour
    )

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_redis_history,
    input_messages_key="input",
    history_messages_key="history",
)
# Now memory persists across server restarts
```

## 10.5 Summary Memory (for Long Conversations)

When conversations get long, token costs explode. Summarize old messages:

```python
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI

memory = ConversationSummaryBufferMemory(
    llm=ChatOpenAI(),
    max_token_limit=500,   # Keep recent messages up to 500 tokens
    # Older messages get summarized automatically
)

# Add messages
memory.save_context(
    {"input": "Hi, I'm working on a machine learning project"},
    {"output": "That sounds interesting! What kind of ML project?"}
)
memory.save_context(
    {"input": "I'm building a recommendation system for movies"},
    {"output": "Great choice! What data are you using?"}
)

# Get the memory
print(memory.load_memory_variables({}))
# {"history": "Human mentioned they're building a movie recommendation system..."}
```

## 10.6 Entity Memory

Track specific entities mentioned in conversation:

```python
from langchain.memory import ConversationEntityMemory
from langchain_openai import ChatOpenAI

memory = ConversationEntityMemory(llm=ChatOpenAI())

memory.save_context(
    {"input": "Alice is my coworker. She's a Python developer."},
    {"output": "Got it!"}
)

print(memory.load_memory_variables({"input": "Tell me about Alice"}))
# {"entities": {"Alice": "coworker, Python developer"}, "history": "..."}
```

## 10.7 Memory in Production: Best Practices

```python
# ✅ Best Practice: Use session-based memory with expiry
from langchain_community.chat_message_histories import SQLChatMessageHistory

def get_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string="sqlite:///chat_history.db"
    )

# ✅ Trim history to avoid token overflow
from langchain_core.messages import trim_messages

trimmer = trim_messages(
    max_tokens=4000,
    strategy="last",        # Keep most recent
    token_counter=ChatOpenAI(model="gpt-4o-mini"),
    include_system=True,    # Always keep system message
    allow_partial=False,
    start_on="human",       # Start trim on human message
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

chain = (
    RunnablePassthrough.assign(history=itemgetter("history") | trimmer)
    | prompt
    | ChatOpenAI()
    | StrOutputParser()
)
```

---

# 11. Document Loaders

## 11.1 What are Document Loaders?

Document Loaders read data from various sources and convert them into `Document` objects that LangChain can process:

```
Source (PDF, URL, DB, API...)
        ↓
  DocumentLoader
        ↓
 [Document, Document, ...]
```

## 11.2 Text & File Loaders

```python
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredFileLoader,
)

# Plain text
loader = TextLoader("path/to/file.txt", encoding="utf-8")
docs = loader.load()

# PDF — creates one Document per page
loader = PyPDFLoader("document.pdf")
docs = loader.load()
print(docs[0].page_content)  # First page text
print(docs[0].metadata)      # {"source": "document.pdf", "page": 0}

# Word document
loader = Docx2txtLoader("document.docx")
docs = loader.load()

# CSV — each row becomes a Document
loader = CSVLoader(
    "data.csv",
    source_column="content",  # Column to use as page_content
    metadata_columns=["date", "author"]
)
docs = loader.load()

# JSON
loader = JSONLoader(
    "data.json",
    jq_schema=".[]",  # jq query to extract content
    text_content=False
)
docs = loader.load()
```

## 11.3 Web Loaders

```python
from langchain_community.document_loaders import (
    WebBaseLoader,
    RecursiveUrlLoader,
    GitbookLoader,
    SitemapLoader,
)
import bs4

# Single URL
loader = WebBaseLoader(
    "https://langchain.readthedocs.io/en/latest/",
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(
            class_=("post-content", "post-title")  # Only extract these classes
        )
    }
)
docs = loader.load()

# Multiple URLs
loader = WebBaseLoader(
    ["https://example.com/page1", "https://example.com/page2"]
)
docs = loader.load()

# Recursive crawling
loader = RecursiveUrlLoader(
    "https://docs.langchain.com",
    max_depth=2,        # How deep to crawl
    use_async=True,
)
docs = loader.load()
```

## 11.4 Database Loaders

```python
from langchain_community.document_loaders import (
    SQLDatabaseLoader,
)
from langchain_community.utilities import SQLDatabase

# Load from SQL database
db = SQLDatabase.from_uri("sqlite:///mydb.db")
loader = SQLDatabaseLoader(
    "SELECT title, content, date FROM articles WHERE category = 'tech'",
    db=db,
    page_content_columns=["content"],  # Use this as page_content
    metadata_columns=["title", "date"]
)
docs = loader.load()
```

## 11.5 Cloud Storage Loaders

```python
# AWS S3
from langchain_community.document_loaders import S3FileLoader, S3DirectoryLoader

loader = S3FileLoader("my-bucket", "path/to/file.pdf")
docs = loader.load()

loader = S3DirectoryLoader("my-bucket", prefix="documents/")
docs = loader.load()

# Google Drive
from langchain_community.document_loaders import GoogleDriveLoader

loader = GoogleDriveLoader(
    folder_id="1abc...",
    recursive=True,
    file_types=["document", "sheet"],
)
docs = loader.load()
```

## 11.6 Lazy Loading (Memory Efficient)

```python
# For large document sets, load lazily to save memory
loader = PyPDFLoader("huge_document.pdf")

# lazy_load returns a generator
for doc in loader.lazy_load():
    process(doc)  # Process one at a time
    # Document is garbage collected after use
```

## 11.7 Custom Document Loader

```python
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from typing import Iterator
import requests

class HackerNewsLoader(BaseLoader):
    """Load top stories from Hacker News"""
    
    def __init__(self, num_stories: int = 10):
        self.num_stories = num_stories
    
    def lazy_load(self) -> Iterator[Document]:
        # Get top story IDs
        ids = requests.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json"
        ).json()[:self.num_stories]
        
        for story_id in ids:
            story = requests.get(
                f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
            ).json()
            
            yield Document(
                page_content=story.get("text", story.get("title", "")),
                metadata={
                    "title": story.get("title"),
                    "url": story.get("url"),
                    "score": story.get("score"),
                    "source": "hackernews"
                }
            )

# Use it
loader = HackerNewsLoader(num_stories=5)
docs = loader.load()
```

---

# 12. Text Splitters

## 12.1 Why Split Text?

Problems if you don't split:
1. **Context window overflow** — Documents often exceed model limits
2. **Retrieval quality** — Small, focused chunks return more relevant results
3. **Cost** — Smaller chunks = fewer tokens = lower cost

```
Full Document (50,000 words)
         ↓  Split
[Chunk 1] [Chunk 2] [Chunk 3] ... [Chunk N]
(~500 words each)
         ↓  Embed & Store
Vector Database
         ↓  Retrieve
Relevant Chunk(s) → Feed to LLM
```

## 12.2 RecursiveCharacterTextSplitter (Recommended)

The most widely used — tries to split on natural boundaries:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Maximum characters per chunk
    chunk_overlap=200,      # Characters to overlap between chunks
    length_function=len,    # How to measure length
    # Tries these separators in order:
    separators=["\n\n", "\n", " ", ""]
    # \n\n = paragraphs (preferred)
    # \n   = lines
    # " "  = words
    # ""   = characters (last resort)
)

text = """Chapter 1: Introduction

This is the first paragraph of the chapter. It contains important information
about the topic at hand.

This is the second paragraph. It continues the discussion.

Chapter 2: Deep Dive

This chapter goes into more detail..."""

chunks = splitter.split_text(text)
print(f"Number of chunks: {len(chunks)}")
print(f"First chunk:\n{chunks[0]}")
print(f"Chunk size: {len(chunks[0])} chars")

# Split Documents (maintains metadata)
docs = splitter.split_documents(raw_documents)
# Each doc keeps the metadata from its parent
```

## 12.3 Choosing Chunk Size

```
Use Case                  | chunk_size | chunk_overlap
─────────────────────────────────────────────────────
Q&A over docs             | 500-1000   | 100-200
Summarization             | 2000-4000  | 200-400
Code documentation        | 500-1500   | 50-200
Books/long narrative      | 1000-2000  | 200-500
Legal/technical docs      | 500-800    | 150-250
```

**Rule of thumb:** The chunk should contain enough context to answer a question on its own, but not so much that retrieval becomes unfocused.

## 12.4 Specialized Text Splitters

```python
# Token-based (more accurate for LLM context limits)
from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=512,       # 512 tokens (not characters!)
    chunk_overlap=50,
    model_name="gpt-4",  # Use this model's tokenizer
)

# Markdown-aware (respects headers)
from langchain_text_splitters import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header1"),
        ("##", "Header2"),
        ("###", "Header3"),
    ]
)

markdown = """# Chapter 1
Content here

## Section 1.1
More content

## Section 1.2
Even more content"""

chunks = splitter.split_text(markdown)
# Each chunk includes the header hierarchy in metadata
print(chunks[0].metadata)  # {"Header1": "Chapter 1", "Header2": "Section 1.1"}

# Python code-aware
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=100,
)
# Knows to split on class/function boundaries

# HTML-aware
splitter = RecursiveCharacterTextSplitter.from_language(language=Language.HTML)
```

## 12.5 Semantic Text Splitting

Split based on meaning rather than character count:

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# Splits where there's a semantic "break"
splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # or "standard_deviation"
    breakpoint_threshold_amount=95,           # 95th percentile difference
)

text = """I love Python programming. It's clean and readable.
Python has a great ecosystem of libraries.

The economy is doing well this quarter. 
Stocks are up 5% year-over-year.

Dogs are wonderful pets. They're loyal and friendly."""

chunks = splitter.split_text(text)
# Will likely create 3 chunks (programming, economy, dogs)
# because each topic is semantically distinct
```

---

# 13. Embeddings

## 13.1 What are Embeddings?

Embeddings are **dense numerical representations** of text. They capture semantic meaning in a vector (list of numbers):

```
"The cat sat on the mat"  →  [0.23, -0.11, 0.89, 0.34, ...]  (1536 numbers)
"A feline rested on a rug" →  [0.24, -0.10, 0.87, 0.35, ...]  (1536 numbers)
"The stock market crashed"  →  [-0.45, 0.78, -0.12, 0.01, ...]  (1536 numbers)
```

Similar meaning → similar vectors (close in vector space)  
Different meaning → different vectors (far in vector space)

This enables **semantic search**: find documents by meaning, not just keywords.

## 13.2 How Embeddings Work

```
             High-dimensional space (1536 dims)
             
             ● "dog"
         ● "puppy"           
    ● "cat"  ● "kitten"
                                ● "car"
                         ● "vehicle" ● "truck"
                    ● "Apple Inc" ● "Microsoft"
                         ● "stock price"
```

Words/sentences with similar meaning cluster together.

## 13.3 OpenAI Embeddings

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",   # Cheaper, 1536 dims
    # model="text-embedding-3-large", # Better quality, 3072 dims
    # model="text-embedding-ada-002", # Legacy
)

# Embed a single text
vector = embeddings.embed_query("What is machine learning?")
print(len(vector))   # 1536
print(vector[:5])    # [0.023, -0.156, 0.089, ...]

# Embed multiple texts (for storage)
texts = [
    "Machine learning is a subset of AI",
    "Python is a programming language",
    "The weather is nice today",
]
vectors = embeddings.embed_documents(texts)
print(len(vectors))     # 3
print(len(vectors[0]))  # 1536
```

## 13.4 Hugging Face Embeddings (Free, Local)

```python
from langchain_huggingface import HuggingFaceEmbeddings

# Runs locally, no API key needed
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    # Other great options:
    # "sentence-transformers/all-mpnet-base-v2"  (better quality)
    # "BAAI/bge-small-en-v1.5"                  (fast, good quality)
    # "thenlper/gte-large"                       (large, high quality)
)

vector = embeddings.embed_query("Hello world")
print(len(vector))  # 384 (smaller than OpenAI)
```

## 13.5 Ollama Embeddings (Free, Local)

```python
from langchain_ollama import OllamaEmbeddings

# First: ollama pull nomic-embed-text
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector = embeddings.embed_query("What is LangChain?")
```

## 13.6 Cosine Similarity

How to measure if two embeddings are similar:

```python
import numpy as np

def cosine_similarity(v1, v2):
    """Returns 1.0 for identical meaning, 0 for unrelated, -1 for opposite"""
    v1, v2 = np.array(v1), np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

embeddings = OpenAIEmbeddings()

v1 = embeddings.embed_query("I love programming")
v2 = embeddings.embed_query("Coding is my passion")
v3 = embeddings.embed_query("The weather is warm")

print(cosine_similarity(v1, v2))  # ~0.92 (very similar!)
print(cosine_similarity(v1, v3))  # ~0.45 (not similar)
```

## 13.7 Choosing an Embedding Model

```
Model                                  | Dims  | Quality | Speed  | Cost
────────────────────────────────────────────────────────────────────────
text-embedding-3-small (OpenAI)        | 1536  | ★★★★☆  | Fast   | $0.02/M tokens
text-embedding-3-large (OpenAI)        | 3072  | ★★★★★  | Medium | $0.13/M tokens  
all-MiniLM-L6-v2 (HuggingFace)        | 384   | ★★★☆☆  | Fast   | Free (local)
all-mpnet-base-v2 (HuggingFace)        | 768   | ★★★★☆  | Medium | Free (local)
BAAI/bge-large-en-v1.5 (HuggingFace)  | 1024  | ★★★★★  | Slow   | Free (local)
nomic-embed-text (Ollama)              | 768   | ★★★★☆  | Fast   | Free (local)
```

---

# 14. Vector Stores

## 14.1 What is a Vector Store?

A vector store is a **database specialized for storing and searching vectors** (embeddings). Unlike a regular database that does exact matching, a vector store does **approximate nearest neighbor (ANN) search** — finding the most similar vectors.

```
Vector Store Operations:
├── add_documents(docs)    — Embed and store
├── similarity_search(q)   — Find most similar docs
├── similarity_search_with_score(q) — Find similar + similarity score
├── max_marginal_relevance_search(q) — Diverse + relevant results
└── delete(ids)            — Remove documents
```

## 14.2 FAISS (In-Memory, Local)

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Install: pip install faiss-cpu

embeddings = OpenAIEmbeddings()

# Create from documents
docs = [
    Document(page_content="LangChain is a framework for LLMs"),
    Document(page_content="Python is a programming language"),
    Document(page_content="Vector databases store embeddings"),
    Document(page_content="Transformers are neural network architectures"),
]

vectorstore = FAISS.from_documents(docs, embeddings)

# Similarity search
results = vectorstore.similarity_search("What is LangChain?", k=2)
for doc in results:
    print(doc.page_content)
# "LangChain is a framework for LLMs"
# "Vector databases store embeddings"

# Search with scores (lower = more similar in L2, higher in cosine)
results = vectorstore.similarity_search_with_score("What is LangChain?", k=2)
for doc, score in results:
    print(f"Score: {score:.3f} | {doc.page_content}")

# Save and load
vectorstore.save_local("my_vectorstore")
loaded = FAISS.load_local(
    "my_vectorstore", 
    embeddings, 
    allow_dangerous_deserialization=True
)

# Add more documents later
vectorstore.add_documents([Document(page_content="New document here")])
```

## 14.3 Chroma (Persistent, Local)

```python
from langchain_chroma import Chroma

# Install: pip install chromadb langchain-chroma

# Create persistent vectorstore
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db",  # Saves to disk
    collection_name="my_docs",
)

# Load existing
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="my_docs",
)

# Search
results = vectorstore.similarity_search("LLM frameworks", k=3)

# Delete by ID
vectorstore.delete(ids=["id1", "id2"])
```

## 14.4 Pinecone (Cloud, Production-Grade)

```python
from langchain_pinecone import PineconeVectorStore
import pinecone

# Install: pip install langchain-pinecone pinecone-client

# Initialize Pinecone (create index first in Pinecone console)
vectorstore = PineconeVectorStore.from_documents(
    docs,
    embeddings,
    index_name="my-index",
    # PINECONE_API_KEY env var
)

# Or connect to existing
vectorstore = PineconeVectorStore(
    index_name="my-index",
    embedding=embeddings,
)

results = vectorstore.similarity_search("query here", k=5)
```

## 14.5 Qdrant (Self-hosted or Cloud)

```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Install: pip install langchain-qdrant qdrant-client

# In-memory
client = QdrantClient(":memory:")

# Local persistent
client = QdrantClient(path="./qdrant_data")

# Remote
client = QdrantClient(url="http://localhost:6333")

vectorstore = QdrantVectorStore.from_documents(
    docs,
    embeddings,
    client=client,
    collection_name="my_collection",
)
```

## 14.6 PGVector (PostgreSQL with Vector Extension)

```python
from langchain_postgres import PGVector

# Install: pip install langchain-postgres psycopg

CONNECTION_STRING = "postgresql+psycopg://user:password@localhost:5432/mydb"

vectorstore = PGVector.from_documents(
    docs,
    embeddings,
    collection_name="langchain_test",
    connection=CONNECTION_STRING,
)
```

## 14.7 Choosing a Vector Store

```
Store        | Deployment  | Scale       | Best For
─────────────────────────────────────────────────────────
FAISS        | In-memory   | Small-Med   | Prototyping, no setup
Chroma       | Local       | Small-Med   | Dev, small production
Qdrant       | Self-hosted | Large       | Production, full control
Pinecone     | Cloud       | Unlimited   | Production, no ops
PGVector     | Postgres    | Med-Large   | Already using PostgreSQL
Weaviate     | Self/Cloud  | Large       | Hybrid search needs
Redis        | Self/Cloud  | Large       | Low latency, existing Redis
```

## 14.8 MMR Search (Diversity in Results)

```python
# Maximal Marginal Relevance: 
# Returns relevant results that are ALSO diverse (not just top similar)
results = vectorstore.max_marginal_relevance_search(
    "Tell me about Python",
    k=4,            # Return 4 results
    fetch_k=20,     # First fetch 20, then pick diverse 4
    lambda_mult=0.5 # 0 = max diversity, 1 = max relevance
)
```

---

# 15. Retrievers

## 15.1 What is a Retriever?

A Retriever is an interface that takes a **string query** and returns a **list of Documents**. It's more flexible than a vector store — can combine multiple search strategies.

```python
# All retrievers have the same interface
retriever.invoke("What is machine learning?")
# Returns: [Document, Document, Document, ...]
```

## 15.2 VectorStoreRetriever

```python
# Basic: Get top-k similar docs
retriever = vectorstore.as_retriever(
    search_type="similarity",       # Default
    search_kwargs={"k": 5}          # Return 5 docs
)

# MMR: Get relevant AND diverse docs
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
)

# Similarity score threshold: Only return docs above a similarity score
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7, "k": 5}
)

# Use the retriever
docs = retriever.invoke("What is deep learning?")
```

## 15.3 MultiQueryRetriever

Generates multiple phrasings of the query to retrieve more relevant docs:

```python
from langchain.retrievers import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=ChatOpenAI(temperature=0)
)

# For query "What is ML?", the LLM generates:
# 1. "What is ML?"
# 2. "Define machine learning"
# 3. "Explain the concept of machine learning"
# Retrieves docs for all 3, deduplicates, returns combined results

docs = retriever.invoke("What is ML?")
```

## 15.4 ContextualCompressionRetriever

Retrieve docs, then compress to only keep relevant parts:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Compressor: Extracts only the relevant parts of each doc
compressor = LLMChainExtractor.from_llm(ChatOpenAI())

retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
)

docs = retriever.invoke("What does LangChain do?")
# Returns compressed docs — only the sentences relevant to the query
```

## 15.5 BM25Retriever (Keyword Search)

Traditional keyword-based search (fast, no embeddings needed):

```python
from langchain_community.retrievers import BM25Retriever

# Install: pip install rank_bm25

retriever = BM25Retriever.from_documents(docs, k=5)
results = retriever.invoke("langchain framework")
```

## 15.6 EnsembleRetriever (Hybrid Search)

Combine keyword and semantic search:

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Create both retrievers
bm25_retriever = BM25Retriever.from_documents(docs, k=5)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Combine them
ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]  # Equal weight; adjust as needed
)

# Gets results from both, combines via Reciprocal Rank Fusion
results = ensemble.invoke("machine learning frameworks")
```

## 15.7 SelfQueryRetriever (Metadata Filtering)

The LLM generates structured queries with metadata filters:

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# Define the metadata schema
metadata_field_info = [
    AttributeInfo(name="genre", description="Movie genre", type="string"),
    AttributeInfo(name="year", description="Release year", type="integer"),
    AttributeInfo(name="rating", description="IMDB rating 1-10", type="float"),
]

retriever = SelfQueryRetriever.from_llm(
    llm=ChatOpenAI(),
    vectorstore=vectorstore,
    document_contents="Movie descriptions",
    metadata_field_info=metadata_field_info,
)

# The LLM translates this natural language query into:
# semantic query: "action movies" + filter: year >= 2020 AND rating > 8
results = retriever.invoke("Good action movies from the last 5 years with high ratings")
```

## 15.8 ParentDocumentRetriever

Store small chunks for retrieval but return the full parent document:

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# Small chunks for retrieval (precise matching)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# Larger chunks to return as context
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

# Storage
docstore = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# Add documents
retriever.add_documents(docs)

# Search finds the small chunk, returns the large parent
results = retriever.invoke("What is LangChain?")
# Returns: full 2000-char parent, not the 400-char child
```

---

# 16. RAG Systems

## 16.1 What is RAG?

**Retrieval-Augmented Generation (RAG)** combines:
1. **Retrieval**: Find relevant information from a knowledge base
2. **Augmentation**: Add that information to the prompt
3. **Generation**: LLM generates a response grounded in that information

```
┌─────────────────────────────────────────────────────────┐
│                        RAG FLOW                          │
│                                                          │
│  Question ──→ [Retriever] ──→ Relevant Docs              │
│      │                              │                    │
│      └──────────────────────────────┼──→ [Prompt] ──→ LLM │
│                                     │         ↑          │
│                               "Use these docs            │
│                                to answer the             │
│                                   question"              │
└─────────────────────────────────────────────────────────┘
```

## 16.2 Why RAG?

Problems RAG solves:
- **Outdated knowledge**: LLMs have a training cutoff. RAG can use live data.
- **Hallucination**: LLMs make things up. Grounding in real docs reduces this.
- **Private data**: LLMs don't know your company's docs. RAG does.
- **Cost**: Fine-tuning is expensive. RAG is cheap.
- **Traceability**: Can cite sources for answers.

## 16.3 Basic RAG Pipeline

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ─── Step 1: Load Documents ───────────────────────────────
loader = WebBaseLoader("https://en.wikipedia.org/wiki/LangChain")
raw_docs = loader.load()

# ─── Step 2: Split Documents ─────────────────────────────
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(raw_docs)

# ─── Step 3: Embed & Store ───────────────────────────────
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ─── Step 4: Build RAG Chain ──────────────────────────────
rag_prompt = ChatPromptTemplate.from_template("""
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, say that you don't know.
Keep the answer concise.

Context:
{context}

Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,  # Retrieve + format
        "question": RunnablePassthrough()    # Pass question through
    }
    | rag_prompt
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# ─── Step 5: Use It ───────────────────────────────────────
answer = rag_chain.invoke("What is LangChain used for?")
print(answer)
```

## 16.4 RAG with Source Citations

```python
from langchain_core.runnables import RunnableParallel

# Return both answer and source documents
rag_chain_with_sources = RunnableParallel(
    answer=rag_chain,
    context=retriever | format_docs,
)

# Alternative: return source docs
rag_chain_with_docs = (
    RunnableParallel(context=retriever, question=RunnablePassthrough())
    .assign(answer=rag_prompt | ChatOpenAI() | StrOutputParser())
)

result = rag_chain_with_docs.invoke("What is LangChain?")
print(result["answer"])   # The answer
print(result["context"])  # The Documents used

# Format sources nicely
sources = [doc.metadata.get("source", "Unknown") for doc in result["context"]]
print(f"Sources: {', '.join(set(sources))}")
```

## 16.5 Conversational RAG

RAG with memory — reference previous conversation:

```python
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

llm = ChatOpenAI(model="gpt-4o-mini")

# Step 1: Reformulate question considering history
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given a chat history and the latest user question 
    which might reference context in the chat history, formulate a 
    standalone question which can be understood without the chat history.
    Do NOT answer the question, just reformulate it if needed."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Step 2: Answer using retrieved context
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using this context:\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Use with memory
from langchain_core.messages import HumanMessage, AIMessage

chat_history = []

question1 = "What is LangChain?"
r1 = rag_chain.invoke({"input": question1, "chat_history": chat_history})
chat_history.extend([HumanMessage(content=question1), AIMessage(content=r1["answer"])])

question2 = "Who created it?"  # "it" refers to LangChain from context
r2 = rag_chain.invoke({"input": question2, "chat_history": chat_history})
print(r2["answer"])  # Correctly answers about LangChain's creator
```

## 16.6 Advanced RAG Techniques

### Reranking
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank  # pip install langchain-cohere

# Reranker: Re-scores retrieved docs for better relevance
reranker = CohereRerank(model="rerank-english-v3.0", top_n=3)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 10})
)
# Retrieves 10, reranks, returns best 3
```

### HyDE (Hypothetical Document Embeddings)
```python
# Generate a hypothetical answer first, embed that for retrieval
# Often improves retrieval quality

hyde_prompt = ChatPromptTemplate.from_template(
    "Write a short paragraph that would answer this question: {question}"
)

hyde_chain = hyde_prompt | ChatOpenAI() | StrOutputParser()

def hyde_retriever(question):
    # Generate hypothetical answer
    hypothetical = hyde_chain.invoke({"question": question})
    # Use hypothetical answer as the search query
    return retriever.invoke(hypothetical)
```

### Multi-Vector Retrieval
```python
# Store multiple representations of each document:
# - The chunk itself
# - A summary of the chunk
# - Hypothetical questions the chunk answers

# Retrieve by any of these, but return the full doc

from langchain.storage import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
import uuid

store = InMemoryByteStore()
id_key = "doc_id"

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)
```

## 16.7 RAG Evaluation Metrics

```
Metric              | What it measures
────────────────────────────────────────────────────────
Context Relevance   | Are the retrieved docs relevant to the question?
Answer Faithfulness | Is the answer grounded in the retrieved context?
Answer Relevance    | Does the answer actually answer the question?
Context Recall      | Did we retrieve all the needed information?
```

## 16.8 Complete RAG System with Best Practices

```python
"""
Production-ready RAG system with:
- Hybrid search (keyword + semantic)
- Reranking
- Source citations
- Conversation history
- Streaming
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

class ProductionRAG:
    def __init__(self, docs):
        self.llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
        self.embeddings = OpenAIEmbeddings()
        
        # Build hybrid retriever
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        self.bm25 = BM25Retriever.from_documents(docs, k=5)
        self.ensemble = EnsembleRetriever(
            retrievers=[self.bm25, self.vectorstore.as_retriever(search_kwargs={"k": 5})],
            weights=[0.4, 0.6]
        )
        
        self.chain = self._build_chain()
    
    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant. Use the context below to answer.
Always cite your sources. If you don't know, say so.

Context:
{context}"""),
            MessagesPlaceholder("history"),
            ("human", "{question}")
        ])
        
        return (
            RunnableParallel(
                context=self.ensemble | self._format_docs,
                question=RunnablePassthrough(),
                history=RunnablePassthrough(),
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _format_docs(self, docs):
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", f"Doc {i+1}")
            formatted.append(f"[{source}]: {doc.page_content}")
        return "\n\n".join(formatted)
    
    def query(self, question: str, history: list = None):
        return self.chain.invoke({
            "question": question,
            "history": history or []
        })
    
    def stream(self, question: str, history: list = None):
        for token in self.chain.stream({
            "question": question,
            "history": history or []
        }):
            yield token
```
---

# 17. Tools & Toolkits

## 17.1 What are Tools?

Tools are **functions that agents can call** to interact with the world. They let an LLM do things it can't do alone:

```
Without Tools:  LLM can only generate text
With Tools:     LLM can search the web, run code, query databases,
                send emails, call APIs, read files, and more
```

A Tool has:
- **name**: Identifier the LLM uses to call it
- **description**: What the tool does (LLM reads this to decide when to use it)
- **function**: The actual Python code that runs
- **args_schema**: Pydantic schema defining expected arguments

## 17.2 Creating Tools

### Method 1: @tool decorator (Simplest)
```python
from langchain_core.tools import tool

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together. Use when you need to perform addition."""
    return a + b

@tool
def get_word_length(word: str) -> int:
    """Returns the number of characters in a word."""
    return len(word)

# Inspect the tool
print(add_numbers.name)         # "add_numbers"
print(add_numbers.description)  # "Add two numbers together..."
print(add_numbers.args)         # {"a": {"type": "integer"}, "b": {"type": "integer"}}

# Call directly
result = add_numbers.invoke({"a": 5, "b": 3})
print(result)  # 8
```

### Method 2: StructuredTool with Pydantic Schema
```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="The search query to look up")
    num_results: int = Field(default=5, description="Number of results to return")

def web_search(query: str, num_results: int = 5) -> str:
    """Searches the web and returns results."""
    # Real implementation would call a search API
    return f"Search results for '{query}': [result1, result2, ...]"

search_tool = StructuredTool.from_function(
    func=web_search,
    name="web_search",
    description="Search the internet for current information",
    args_schema=SearchInput,
    return_direct=False,  # If True, return tool output directly to user
)
```

### Method 3: BaseTool class (Most Flexible)
```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type
import requests

class WeatherInput(BaseModel):
    city: str = Field(description="The city name to get weather for")
    units: str = Field(default="metric", description="celsius or fahrenheit")

class WeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Get current weather for a city. Use when asked about weather."
    args_schema: Type[BaseModel] = WeatherInput
    
    def _run(self, city: str, units: str = "metric") -> str:
        """Synchronous execution"""
        # Call weather API
        api_key = os.getenv("WEATHER_API_KEY")
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "units": units}
        
        try:
            resp = requests.get(url, params=params)
            data = resp.json()
            temp = data["main"]["temp"]
            desc = data["weather"][0]["description"]
            return f"Weather in {city}: {temp}°, {desc}"
        except Exception as e:
            return f"Error getting weather: {str(e)}"
    
    async def _arun(self, city: str, units: str = "metric") -> str:
        """Async execution"""
        import aiohttp
        async with aiohttp.ClientSession() as session:
            # async implementation
            pass

weather_tool = WeatherTool()
result = weather_tool.invoke({"city": "London", "units": "metric"})
```

## 17.3 Built-in Tools

```python
# DuckDuckGo Search (free, no API key)
from langchain_community.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()
result = search.invoke("Latest AI news 2024")

# Wikipedia
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
result = wiki.invoke("Albert Einstein")

# Python REPL (execute Python code)
from langchain_experimental.tools import PythonREPLTool
python_repl = PythonREPLTool()
result = python_repl.invoke("print(2 ** 10)")  # "1024"

# Shell (run bash commands) — USE WITH CAUTION!
from langchain_community.tools import ShellTool
shell = ShellTool()
result = shell.invoke({"commands": ["ls -la", "echo hello"]})

# Tavily Search (best search tool, needs API key)
from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults(max_results=3)
result = search.invoke("current OpenAI CEO")

# ArXiv (research papers)
from langchain_community.tools import ArxivQueryRun
arxiv = ArxivQueryRun()
result = arxiv.invoke("transformer architecture attention mechanism")
```

## 17.4 File Tools

```python
from langchain_community.tools.file_management import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
    CopyFileTool,
    DeleteFileTool,
    MoveFileTool,
)
from langchain_community.agent_toolkits import FileManagementToolkit

# Create a toolkit (restricted to a directory for safety)
toolkit = FileManagementToolkit(root_dir="/tmp/agent_workspace")
tools = toolkit.get_tools()
# Returns: [ReadFileTool, WriteFileTool, ListDirectoryTool, ...]

# Use individually
read_tool = ReadFileTool(root_dir="/tmp/agent_workspace")
content = read_tool.invoke({"file_path": "notes.txt"})

write_tool = WriteFileTool(root_dir="/tmp/agent_workspace")
write_tool.invoke({"file_path": "output.txt", "text": "Hello from agent!"})
```

## 17.5 Database Tools

```python
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import (
    QuerySQLDataBaseTool,
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
)

db = SQLDatabase.from_uri("sqlite:///mydatabase.db")

# Tools to give an agent SQL access
query_tool = QuerySQLDataBaseTool(db=db)
info_tool = InfoSQLDatabaseTool(db=db)
list_tool = ListSQLDatabaseTool(db=db)

# List all tables
tables = list_tool.invoke("")
# Get info about a table
info = info_tool.invoke("users")
# Run a query
result = query_tool.invoke("SELECT COUNT(*) FROM users")
```

## 17.6 Toolkits

A Toolkit is a pre-packaged collection of related tools:

```python
# GitHub Toolkit
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper

github = GitHubAPIWrapper()
toolkit = GitHubToolkit.from_github_api_wrapper(github)
tools = toolkit.get_tools()
# read_file, create_file, update_file, list_open_pull_requests, etc.

# Pandas DataFrame Toolkit
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd

df = pd.read_csv("sales_data.csv")
agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0),
    df,
    verbose=True,
    allow_dangerous_code=True
)
agent.invoke("What is the average revenue by region?")

# SQL Database Toolkit
from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI())
tools = toolkit.get_tools()  # query, schema info, table list tools
```

## 17.7 Tool Error Handling

```python
from langchain_core.tools import tool, ToolException

@tool
def divide(numerator: float, denominator: float) -> float:
    """Divide two numbers. Raises error if denominator is zero."""
    if denominator == 0:
        raise ToolException("Cannot divide by zero!")
    return numerator / denominator

# Tool with handle_tool_error
from langchain_core.tools import StructuredTool

safe_divide = StructuredTool.from_function(
    func=divide.func,
    name="divide",
    description="Divide two numbers",
    handle_tool_error=True,  # Catches ToolException, passes message to agent
)

# Custom error handling
safe_divide = StructuredTool.from_function(
    func=divide.func,
    name="divide",
    description="Divide two numbers",
    handle_tool_error=lambda e: f"Tool failed: {str(e)}. Try different inputs.",
)
```

---

# 18. Agents

## 18.1 What is an Agent?

An Agent is a system where the **LLM decides which actions to take** in a loop:

```
┌─────────────────────────────────────────────────────────┐
│                    AGENT LOOP                            │
│                                                          │
│  Input ──→ LLM ──→ "I need to search the web"           │
│               ↓                                          │
│         Tool Call: search("latest AI news")              │
│               ↓                                          │
│         Tool Result: "OpenAI released..."                │
│               ↓                                          │
│         LLM ──→ "I need more info about OpenAI"          │
│               ↓                                          │
│         Tool Call: search("OpenAI company info")         │
│               ↓                                          │
│         Tool Result: "OpenAI is..."                      │
│               ↓                                          │
│         LLM ──→ "I have enough to answer" → Final Answer  │
└─────────────────────────────────────────────────────────┘
```

## 18.2 ReAct Agent (Most Common)

ReAct = **Re**ason + **Act**. The model reasons, then acts, then observes, repeat.

```python
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

# Initialize tools
search = DuckDuckGoSearchRun()
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [search, wiki]

# Get the ReAct prompt from LangChain hub
# (or create your own — see below)
prompt = hub.pull("hwchase17/react")

# Create agent (the LLM + prompt + tools)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_react_agent(llm, tools, prompt)

# Create executor (runs the agent loop)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,           # Print reasoning steps
    max_iterations=10,      # Max loops before stopping
    handle_parsing_errors=True,  # Handle LLM format errors gracefully
)

# Run the agent
result = agent_executor.invoke({
    "input": "Who is the current CEO of Anthropic, and what did they study at university?"
})
print(result["output"])
```

**What verbose output looks like:**
```
> Entering new AgentExecutor chain...
Thought: I need to find out who the CEO of Anthropic is.
Action: duckduckgo_search
Action Input: Anthropic CEO 2024
Observation: Dario Amodei is the CEO of Anthropic...
Thought: Now I need to find what Dario studied at university.
Action: wikipedia
Action Input: Dario Amodei
Observation: Dario Amodei studied physics at Caltech...
Thought: I have all the information needed.
Final Answer: Dario Amodei is the CEO of Anthropic. He studied physics at Caltech...
```

## 18.3 Tool-Calling Agent (Modern, Recommended)

Uses native function/tool calling built into modern LLMs:

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Input should be a valid Python math expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, 
                     {"abs": abs, "round": round, "min": min, "max": max})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool  
def get_current_date() -> str:
    """Get today's date."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")

tools = [calculator, get_current_date, DuckDuckGoSearchRun()]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use tools when needed."),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),  # Required: where tool results go
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = executor.invoke({"input": "What is 15% of 3847, and what's today's date?"})
print(result["output"])
```

## 18.4 Agent with Memory

```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

agent_with_memory = RunnableWithMessageHistory(
    executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

config = {"configurable": {"session_id": "user_42"}}

r1 = agent_with_memory.invoke(
    {"input": "My name is Alice and I'm a data scientist."},
    config=config
)

r2 = agent_with_memory.invoke(
    {"input": "What's my name and profession?"},
    config=config
)
print(r2["output"])  # "Your name is Alice and you're a data scientist."
```

## 18.5 Streaming Agent Output

```python
# Stream token by token
for event in executor.stream({"input": "What is the capital of France?"}):
    # Event types: on_chain_start, on_tool_start, on_tool_end, on_chain_end
    if "output" in event:
        print(event["output"], end="", flush=True)

# Stream with full event details
async for event in executor.astream_events(
    {"input": "Search for the latest AI news"},
    version="v2"
):
    kind = event["event"]
    if kind == "on_tool_start":
        print(f"\n🔧 Using tool: {event['name']}")
    elif kind == "on_tool_end":
        print(f"✅ Tool result: {event['data']['output'][:100]}...")
    elif kind == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="", flush=True)
```

## 18.6 Custom Agent from Scratch

```python
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Custom prompt with tool descriptions
SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

Available tools:
{tools}

Tool names: {tool_names}

To use a tool, respond in this format:
Thought: [your reasoning]
Action: [tool name]
Action Input: [tool input]

After getting the tool results, continue reasoning until you have the final answer.
Then respond:
Thought: I now have the answer
Final Answer: [your answer]"""

# Custom output parser
def parse_output(llm_output: str):
    if "Final Answer:" in llm_output:
        return AgentFinish(
            return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
            log=llm_output,
        )
    
    # Parse action
    action_match = re.search(r"Action: (.*?)[\n]", llm_output)
    input_match = re.search(r"Action Input: (.*?)[\n]", llm_output)
    
    if action_match and input_match:
        action = action_match.group(1).strip()
        action_input = input_match.group(1).strip()
        return AgentAction(tool=action, tool_input=action_input, log=llm_output)
    
    raise ValueError(f"Could not parse output: {llm_output}")
```

## 18.7 Agent Best Practices

```python
# 1. Always set max_iterations to prevent infinite loops
executor = AgentExecutor(agent=agent, tools=tools, max_iterations=10)

# 2. Set the early stopping method
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    early_stopping_method="generate",  # Generate a final answer if max iterations hit
)

# 3. Return intermediate steps for debugging
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    return_intermediate_steps=True,
)
result = executor.invoke({"input": "..."})
for action, observation in result["intermediate_steps"]:
    print(f"Tool: {action.tool}, Input: {action.tool_input}")
    print(f"Result: {observation}\n")

# 4. Good tool descriptions are CRITICAL
@tool
def search(query: str) -> str:
    # ❌ Bad description:
    "Search the web."""
    
    # ✅ Good description:
    """ Search the internet for current, real-time information about any topic.
    Use this when you need information about recent events, current facts,
    or anything that requires up-to-date data.
    Input should be a search query string."""
    pass
```
