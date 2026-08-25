# Chunking Strategy Documentation

## Selected Chunking Method
In this project we used fixed-size text chunking.

## Chunk Size
Each chunk consists of 500 words.

## Chunk Overlap
Each consecutive chunk shares an overlap of 50 words.

## Why These Values Were Chosen
- A 500-word chunk size is large enough to preserve the context of the information and small enough to avoid any unrelated information.
- A 50-word overlap will prevent context loss at chunk boundaries and helps in preserving complete sentence and paragraphs.

## Potential Limitations
- Context Splitting: The model may lose context by cutting sentences or paragraphs in half. For example, splitting a sentence into two chunks might leave the second chunk without enough information for proper interpretation.
- Loss of coherence: Cutting sections of a paragraph arbitrarily may lead to a decrease in comprehension and coherence.
- A chunk can cover various topics if the document shifts between subjects often, and it does not take into account the document's structure like headings or sections.