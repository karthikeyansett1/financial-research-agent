# Phase 2 — Multi-Tool Financial Agent

## What this does
Extends the Phase 1 RAG into an autonomous agent that can reason across multiple 
tools to answer complex financial questions that require both document knowledge 
and live data.

## How the agent works
The agent runs in a loop — it receives a question, decides which tool to call, 
gets the result, then decides whether to call another tool or return a final answer.
Maximum 5 iterations to prevent runaway loops.

## Tools

**search_document:** Queries the FAISS index built in Phase 1. Used for questions 
about the loaded financial document.

**calculate:** Safe eval of mathematical expressions. Used for growth rates, 
margins, ratios — any financial math the agent needs to perform.

**web_search:** DuckDuckGo search for live news and current data not covered 
in the document.

## Architecture decision
Tool selection is driven entirely by the LLM via structured JSON output. 
The agent prompt instructs the model to respond with a JSON object specifying 
the tool and input. This is a lightweight alternative to LangChain agents — 
same reasoning loop, less abstraction, more control.

## Example
Question: "What was Apple's Services revenue growth rate and what is the latest news about Apple?"

Agent reasoning:
1. Calls search_document for Services revenue figures
2. Calls calculate to compute growth rate
3. Calls web_search for current news
4. Returns combined answer grounded in both sources
