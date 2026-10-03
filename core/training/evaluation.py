# core/training/evaluation.py
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

class Evaluator:
    @staticmethod
    def evaluate(y_true, y_pred, target_names=["Normal", "Fighting", "Falling"]):
        """
        Đánh giá chuyên sâu 3 lớp (chuẩn CS406).
        Không chỉ báo cáo 1 chỉ số, mà bóc tách từng class để thấy rõ F1-score của lớp thiểu số.
        """
        # Overall Accuracy
        acc = accuracy_score(y_true, y_pred)

        # Macro average (Tôn trọng các class nhỏ như Falling)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        
        # Per-class metrics
        per_class_p, per_class_r, per_class_f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)

        # Detailed Report String
        report_str = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)

        return {
            "accuracy": acc,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "per_class": {
                "precision": per_class_p.tolist(),
                "recall": per_class_r.tolist(),
                "f1": per_class_f1.tolist(),
                "support": support.tolist()
            },
            "confusion_matrix": cm.tolist(),
            "report_str": report_str
        }

    @staticmethod
    def print_report(eval_results):
        print("\n"+ "="*50)
        print("CLASSIFICATION EVALUATION REPORT")
        print("="*50)
        print(f"Overall Accuracy: {eval_results['accuracy'] * 100:.2f}%")
        print(f"Macro F1-Score:   {eval_results['macro_f1'] * 100:.2f}%\n")
        print("Chi tiết từng lớp (Per-class Metrics):")
        print(eval_results["report_str"])
        print("\nMa trận nhầm lẫn (Confusion Matrix):")
        for row in eval_results["confusion_matrix"]:
            print(row)
        print("="*50 + "\n")