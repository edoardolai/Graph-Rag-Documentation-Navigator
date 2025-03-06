# cli.py
from query_engine import query_codebase


def main():
    print("CodeCarbon Knowledge Graph Explorer")
    print("-----------------------------------")
    print("Ask questions about the CodeCarbon codebase (type 'exit' to quit)")

    while True:
        question = input("\nQuestion: ")
        if question.lower() in ("exit", "quit"):
            break

        try:
            answer = query_codebase(question)
            print("\nAnswer:")
            print(answer["answer"])
            print("Query: \n", answer["query"])
            print("Context: ", answer["context"])
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
