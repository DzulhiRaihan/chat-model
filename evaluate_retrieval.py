import pandas as pd
from vector import vector_store as food_vector_store
from general_data import vector_store as general_vector_store
import numpy as np
from food_ground_truth import food_ground_truth
from general_ground_truth import general_ground_truth

def calculate_precision_recall_at_k(retrieved_docs, relevant_docs, k):
  
    # Take top k retrieved documents
    retrieved_at_k = retrieved_docs[:k]
    
    true_positives = len(set(retrieved_at_k) & set(relevant_docs))
   
    precision = true_positives / k if k > 0 else 0
    
    # Calculate recall@k
    recall = true_positives / len(relevant_docs) if len(relevant_docs) > 0 else 0
    
    return precision, recall

def evaluate_retrieval_system(vector_store, queries, ground_truth, k_values=[1, 3, 5]):
    results = {}
    
    for k in k_values:
        precisions = []
        recalls = []
        
        for query in queries:
            # Get retrieved documents
            retrieved_docs = [doc.metadata['id'] for doc in vector_store.similarity_search(query, k=k)]
            
            # Get relevant documents from ground truth
            relevant_docs = ground_truth.get(query, [])
            
            # Calculate metrics
            precision, recall = calculate_precision_recall_at_k(retrieved_docs, relevant_docs, k)
            
            precisions.append(precision)
            recalls.append(recall)
            
            # Print detailed results for each query
            print(f"\nQuery: {query}")
            print(f"Retrieved IDs: {retrieved_docs}")
            print(f"Relevant IDs: {relevant_docs}")
            print(f"Precision@{k}: {precision:.4f}")
            print(f"Recall@{k}: {recall:.4f}")
        
        # Calculate average metrics
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        
        results[k] = {
            'precision@k': avg_precision,
            'recall@k': avg_recall
        }
    
    return results

def print_evaluation_summary(results, system_name):
    print(f"\n{'='*50}")
    print(f"Evaluation Summary for {system_name}")
    print(f"{'='*50}")
    
    for k, metrics in results.items():
        print(f"\nk={k}:")
        print(f"Average Precision@{k}: {metrics['precision@k']:.4f}")
        print(f"Average Recall@{k}: {metrics['recall@k']:.4f}")

if __name__ == "__main__":
    # Get queries from ground truth
    food_queries = list(food_ground_truth.keys())
    general_queries = list(general_ground_truth.keys())
    
    # Evaluate food vector store
    print("\nEvaluating Food Vector Store")
    food_results = evaluate_retrieval_system(food_vector_store, food_queries, food_ground_truth)
    print_evaluation_summary(food_results, "Food Vector Store")
    
    # Evaluate general vector store
    print("\nEvaluating General Vector Store")
    general_results = evaluate_retrieval_system(general_vector_store, general_queries, general_ground_truth)
    print_evaluation_summary(general_results, "General Vector Store") 