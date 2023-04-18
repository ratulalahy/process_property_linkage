from typing import List
from base_model import BaseModel
from data import CustomDataset

class ComparePerformance:
    def __init__(self, models: List[BaseModel], dataset: CustomDataset):
        self.models = models
        self.dataset = dataset
    
    def compare_metrics(self):
        """
        Compare the performance of the models on the dataset
        """
        for model in self.models:
            # Train the model on the dataset
            model.train(self.dataset.train_data, self.dataset.train_labels)
            
            # Test the model on the validation set
            predictions = model.predict(self.dataset.val_data)
            
            # Print the evaluation metrics for the model
            print(f"Evaluation metrics for model {model.name}:")
            print(f"Accuracy: {model.evaluate_accuracy(predictions, self.dataset.val_labels)}")
            print(f"Precision: {model.evaluate_precision(predictions, self.dataset.val_labels)}")
            print(f"Recall: {model.evaluate_recall(predictions, self.dataset.val_labels)}")
            print(f"F1 score: {model.evaluate_f1_score(predictions, self.dataset.val_labels)}")
            print("------------------------------")
