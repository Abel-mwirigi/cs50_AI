import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    numberofpages = len(corpus)
    prob = dict()

    # Get the set of links from the current page
    links = corpus[page]

    # If the current page has no links, treat it as linking to all pages
    if not links:
        for pages in corpus:
            prob[pages] = 1 / numberofpages
        return prob
    
    for pages in corpus:
        #Base probability from random jump
        prob[pages] = (1 - damping_factor) / numberofpages

        #if page is linked to, add the probability of following the link
        if pages in links:
            prob[pages] += damping_factor / len(links)

    return prob 

    raise NotImplementedError


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # initialize visits counts for each page to 0
    page_counts = { page: 0 for page in corpus }

    current_page = random.choice(list(corpus.keys()))
    page_counts[current_page] += 1

    for _ in range(1, n):
        probabilities = transition_model(corpus, current_page, damping_factor)

        next_page = random.choices(
            population = list(probabilities.keys()),
            weights = list(probabilities.values()),
            k=1
        )[0]

        page_counts[next_page] += 1
        current_page = next_page

    pagerank = {page: count / n for page, count in page_counts.items()}
    return pagerank
    raise NotImplementedError


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    n = len(corpus)
    pagerank = {page: 1 / n for page in corpus}


    converged = False
    while not converged:
        new_ranks = dict()
        converged = True

        for page in corpus:
            if not corpus[page]:
                corpus[page] = set(corpus.keys())

            rank = (1 - damping_factor) / n
            for page_i in corpus:
                if page in corpus[page_i]:
                    rank += damping_factor * (pagerank[page_i] / len(corpus[page_i]))
            new_ranks[page] = rank

            if abs(new_ranks[page] - pagerank[page]) > 0.001:
                converged = False

        pagerank = new_ranks
    return pagerank
    raise NotImplementedError


if __name__ == "__main__":
    main()
