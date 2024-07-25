import json
import os

from PIL import Image
from torch.utils.data import Dataset



class OODCVDataset(Dataset):
    """Class for OOD-CV VQA dataset."""

    def __init__(
            self,
            image_base_dir_path="ood-cv/",
            question_path="ood-cv/template_ood_mcqa_v2.json",
    ):
        self.question_annot = json.load(open(question_path))
        self.image_base_dir_path = image_base_dir_path

    def __len__(self):
        return len(self.question_annot)

    def get_img_path(self, instance):
        return os.path.join(
            self.image_base_dir_path, instance
        )

    def __getitem__(self, idx):
        question = self.question_annot[idx]['question']
        answer = self.question_annot[idx]['answer']
        text_answer = self.question_annot[idx]['text_answer']
        options = self.question_annot[idx]['options']
        situation = self.question_annot[idx]['situation']
        img_path = self.get_img_path(self.question_annot[idx]['image'])

        return {
            "image": img_path,
            "question": question,
            "answer": answer,
            "text_answer": text_answer,
            "options": options,
            "situation": situation
        }

class SketchyDataset(Dataset):
    """Class for OOD-CV VQA dataset."""

    def __init__(
            self,
            image_base_dir_path="/data/datasets",
            question_path="skechydata/template_sketchy_v1.json",
    ):
        self.question_annot = json.load(open(question_path))
        self.image_base_dir_path = image_base_dir_path

    def __len__(self):
        return len(self.question_annot)

    def get_img_path(self, instance):
        return os.path.join(
            self.image_base_dir_path, instance
        )

    def __getitem__(self, idx):
        question = self.question_annot[idx]['question']
        answer = self.question_annot[idx]['answer']
        img_path = self.get_img_path(self.question_annot[idx]['image'])

        return {
            "image": img_path,
            "question": question,
            "answer": answer,
        }