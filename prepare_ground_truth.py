import pandas as pd
from vector import vector_store as food_vector_store
from general_data import vector_store as general_vector_store

def show_documents(vector_store, query, k=5):
    """
    Show documents retrieved for a query and allow user to select relevant ones
    """
    print(f"\nQuery: {query}")
    print("=" * 50)
    
    # Get retrieved documents
    docs = vector_store.similarity_search(query, k=k)
    
    # Show documents and get user input
    relevant_ids = []
    for i, doc in enumerate(docs, 1):
        print(f"\nDocument {i}:")
        print(f"ID: {doc.metadata.get('id', 'N/A')}")
        print(f"Content: {doc.page_content[:200]}...")  # Show first 200 chars
        print("-" * 30)
        
        # Ask user if this document is relevant
        while True:
            response = input("Is this document relevant? (y/n): ").lower()
            if response in ['y', 'n']:
                if response == 'y':
                    relevant_ids.append(doc.metadata.get('id', 'N/A'))
                break
            print("Please enter 'y' or 'n'")
    
    return relevant_ids

def prepare_ground_truth():
    # Food queries
    food_queries = [
        "Beras",
        "Daging sapi",
        "Telur ayam",
        "Sayur bayam",
        "Buah apel"
    ]
    
    # General queries
    general_queries = [
        "Faktor-faktor yang mempengaruhi berat badan",
        "karbohidrat",
        "Pola makan sehat",
        "Menurunkan berat badan",
        "Diet seimbang"
    ]
    
    # Prepare ground truth for food vector store
    print("\nPreparing ground truth for Food Vector Store")
    print("=" * 50)
    food_ground_truth = {}
    for query in food_queries:
        print(f"\nProcessing query: {query}")
        relevant_ids = show_documents(food_vector_store, query)
        food_ground_truth[query] = relevant_ids
    
    # Prepare ground truth for general vector store
    print("\nPreparing ground truth for General Vector Store")
    print("=" * 50)
    general_ground_truth = {}
    for query in general_queries:
        print(f"\nProcessing query: {query}")
        relevant_ids = show_documents(general_vector_store, query)
        general_ground_truth[query] = relevant_ids
    
    # Save ground truth to files
    with open('food_ground_truth.py', 'w') as f:
        f.write("food_ground_truth = {\n")
        for query, ids in food_ground_truth.items():
            f.write(f"    '{query}': {ids},\n")
        f.write("}\n")
    
    with open('general_ground_truth.py', 'w') as f:
        f.write("general_ground_truth = {\n")
        for query, ids in general_ground_truth.items():
            f.write(f"    '{query}': {ids},\n")
        f.write("}\n")
    
    print("\nGround truth has been saved to 'food_ground_truth.py' and 'general_ground_truth.py'")

if __name__ == "__main__":
    prepare_ground_truth() 