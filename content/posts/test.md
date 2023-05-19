+++
title = "RobApp: Meditating with LLMs or how to find relevant information"
author = ["Andres Velasquez"]
draft = false
+++

In Burbeas voice i heard "What would it feel to know that you are ok?", very clearly. I gotta make a meditation teacher i can actually use


## Idea: {#idea}

-   Dont make it a meditation master but a way to help people find information better?


## Worklog: {#worklog}


## Setup {#setup}


### Imports {#imports}

importing and setting up api keys

```python
import os
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import pandas as pd
import tiktoken

# get api key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
```


### Loading file {#loading-file}

We import and read the document to clean it and save metadata from it and for later embedding and use.

```python
reader = PdfReader("texts/2006_0209WorkingandAwakening.pdf")
number_of_pages = len(reader.pages)
text_lines = []
for page in reader.pages:
    ## clean the page numerator
    raw = page.extract_text()
    cleaned = raw.split('\n')[:-1]
    text_lines += cleaned
text_lines[:10]
```


### Extracting metadata {#extracting-metadata}

Here we get the table of contents, the name of the file, and the start of the file

```python
## 1. get the main title
main_title = text_lines[0]
## 2. Get the table of contents
def table_contents(text):
    subtitles = []
    for line in text:
        if '....' in line:
            parts = line.split('.', 1)
            parts[1] = parts[1].lstrip('.').strip()
            parts[0] = parts[0].strip()
            subtitles.append(parts[0])
    return subtitles
table_contents = table_contents(text_lines)
## 3. find the start of the file (end of the table of contents)
def text_start_line(text):
    in_tc = False
    for i, line in enumerate(text):
        if ('....' not in line) and in_tc:
            return i
        elif '....' in line:
            in_tc = True
start = text_start_line(text_lines)
table_contents
```


### Cleaning {#cleaning}

We preserve in the text only whats after the table of contents

```python
text_lines = text_lines[start:]
text_lines[:10]
```

We also have to clean the footnotes. They are usually at the end of the sections.


### Section chunking {#section-chunking}

The documents are from the retreats and the sections contain the individual talks from that retreat. We will break the file into subsection for preparation for further manipulation.

We need to build a dictonary with keys being the subsection title and the value is the text content. We will evaluate everynew line to check if it is a subsection.

```python
def split_text_into_sections(text, table_of_contents):
    sections = {}
    for line in text_lines:
        if line in table_of_contents:
            current_section = line
            sections[current_section] = ''
        elif current_section is not None:
            sections[current_section] += line + '\n'
    return sections
sectioned_text = split_text_into_sections(text_lines, table_contents)
```

lets save this sectiones text to a file in json format

```python
import json

os.makedirs('cleaned_texts', exist_ok=True)
with open('cleaned_texts/sections.json', 'w') as f:
    json.dump(sectioned_text, f, indent=2)
```


## Main shit {#main-shit}

The bot will answear meditation related questions with the help of the context composition. Towards this goal we need to think about how we are going to compose the factual context, how we will implement the process of retrieving information.

The process of retrieving relevant exxcerpts could be done either by doing ****semantic search****, ****human-natural search**** or a combination of the two


### Context composition: {#context-composition}

We will call the part of the prompt composed of the documents excerpts "factual context"

-   How much of the model prompt will be composed of the documents retrieval?
-   Whats going to be the composition of the document based context?
    -   How many excerpts max?
    -   whats going to be the bar for adding context to the prompt.
        We need to be carefull with the quality of the excerpts since models have shown to get derailed with irrelevant information in the prompt.

I think for starters we can create an arbitrary bar for what will be considered a worthy excerpt, this will be a certain treshold of the cosine similarity.
We can then expand then maybe expand the excerpts until they have all the text?, givin that it is relevant.


### Search implementation {#search-implementation}


#### Human/natural search {#human-natural-search}

In this case i want the model to simulate the human though process for lookinf for information in books or a data base.
How do humans look for things?
Lets work on an example:
Lets say i want to look for something rob burbea said about the third jhana. I have access to all the transcriptions of rob retreats and talks.

-   first i will choose the documents i which i might find something
-   This is done one document at a time
-   Then i look for a section that might be relevant
-   Then i start reading that part, if lets say one or two parragraphs look like this is a whole different topic i might just break it and look in other section, or if the section is long enough i might jump to two pages ahead. (theres something like tolerance). Then if i find something cool i might stop and go back and open my context height.
-   Then i might be satisffied, if not i will go back in the chunking to the section part and look for more, if not ill open again and then move to documents.

From the above we can start seeing some patters:

-   Theres are chunking levels where the agent can go up and down (document a to section from document a) and sideways (next section, next parragraph)
-   the agent needs to have a is this what i am looking for?" termomether for all this levels."Is this document related to my search, is this section relate, is this parragraph."
-   there must be a bar for different things;
    -   the agent says, ok this is sufficiently related to what i was looking for, ill stop the search.
    -   the agent gives up and after searching in this context level, or the agent gives up completely

From all this then we see we could map a general agent to search trough texts
We could also maybe join this together by the use of semantical embedding search:
some ideas

-   assign more weight to sections that appeared on the embedding search

And yeah a lot of different interactions

-   Theres something like layers of meaning being discarded. theres a natural chunking of things. Books, then sections, then parragraphs,

theres a difference in looking for context and looking for a particular thing. As we are creating a meditation teacher we need to be able to do both but we will be heavier on the ambigous context search, ie: "how do i deal with loss?". There might not be a single thing where Rob specifically talks about dealing with loss but a lot of differnte similary related things that might be good to contextualize an answear.
Maybe we could tell the agent to make up an answear excatly with this context when instead of regurgitating what it found.
well theres two possibilites then

-   the agent found an answear
-   the agent found context to make an answear
-   the agent couldnt find anything relevant

i think it will be important to me sincere and say that the agent couldnt find anything relevant in the text about it.


#### Semantic search {#semantic-search}

For this method we need to split the text into chunks and then pass it trough an embedding model, then we pass the users inquiry for a transformation process and then we pass this transformed inquiry to the embedding api, then we do some ranking of cosine similarities to get the best excerpts.

<!--list-separator-->

-  Semantic chunking

    How will we chunk the document. This is not a problem where we just need a little fact like "the kign died in 1996", we need more context.
    this is an important problem but for now we can use what openai did in their [wikipedia dataset for the semantic search cookbook example](https://github.com/openai/openai-cookbook/blob/925dd22eea4fe3cc0bfef563b31b3f0e5f9cb433/examples/Embedding_Wikipedia_articles_for_search.ipynb) from open ai.

    -   We need to find a way to check when we have a parragraph and split the sections we have into parragraphs.
    -   We need to make a function to chunk the sections.
    -   We need to prettify the chunks by adding the title of the document and the section name. They do this in openai, i dont know if its necessary.

    This things are not necessarily done in this order

<!--list-separator-->

- <span class="org-todo todo TODO">TODO</span>  make a diagram of the semantic search process
