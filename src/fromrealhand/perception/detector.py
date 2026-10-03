"""Local official Grounding DINO weights; never reads simulation labels."""
import time
import numpy as np


class CupDetector:
    def __init__(self,model_path):
        import torch
        from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
        self.torch=torch; self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.processor=AutoProcessor.from_pretrained(str(model_path),local_files_only=True)
        self.model=AutoModelForZeroShotObjectDetection.from_pretrained(str(model_path),local_files_only=True).to(self.device).eval()

    def detect(self,rgb,threshold=.30):
        from PIL import Image
        start=time.monotonic()
        inputs=self.processor(images=Image.fromarray(rgb),text='a mug.',return_tensors='pt').to(self.device)
        with self.torch.inference_mode(): outputs=self.model(**inputs)
        results=self.processor.post_process_grounded_object_detection(outputs,inputs.input_ids,
            box_threshold=threshold,text_threshold=.25,target_sizes=[rgb.shape[:2]])[0]
        boxes=[]
        for box,score in zip(results['boxes'].cpu().numpy(),results['scores'].cpu().numpy()):
            boxes.append(dict(box_xyxy=box.tolist(),score=float(score),label='mug'))
        return sorted(boxes,key=lambda x:-x['score']),time.monotonic()-start
