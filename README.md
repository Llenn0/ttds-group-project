# GutenTag - A Multi-lingual Search Engine for Project Gutenberg Books.
*where books are Guten-free and searching is a piece of cake\*~*
> * Guten Tag is "Good Day" in Germany
___
This is the back-end of GutenTag, powered by Cloud Run at [this link](https://ttds-gutenberg-fvyohsgcaq-nw.a.run.app/), for front-end please visit https://gutenberg-search-fvyohsgcaq-ez.a.run.app/hello (the `hello` part is necessary in some browsers due to limitations in routing)

Supported search methods:
- **Semantic Search:** a sentence-transformer model is used to embed query text into feature vector and cosine similarity is used to perform an approximated nearest-neighbour search with feature vectors of all English books, results are ranked by cosine similarit
- **Keyword Search:** rank books based on cosine similarity of query text's TF-IDF vector and each book's TF-IDF vector, can be a bit slow (5-10 seconds)
- **Phrase/Proximity Search:** retrieve all books with phrases that match the query text, `tolerance` (in front-end) or `dist` (in POST request to back-end) controls the maximum distance between two words in query that are allowed in retrieval results, 1 means exact match, which isn't recommended except for English-only searches, phrase search uses a positional inverted index stored on Google's FireStore database, so please don't search with phrases that are too long to prevent a timeout
- **Advanced Search:** mirrors Project Gutenberg's advanced search function, allows user to search with author name and title, the matching is exact so if you remember an author only by his first name, don't attempt to guess the last name, but try to remember one of his book titles, order of words doesn't matter for this search
