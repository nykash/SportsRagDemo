from utils.retrieval import retrieve

def test_retrieval():
    results = retrieve("Nishida scores a point")
    print(results)


if __name__ == "__main__":
    test_retrieval()